#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA error: %s (%s:%d)\n", cudaGetErrorString(e),__FILE__,__LINE__); return 2; } } while(0)

__global__ void async_copy(const float* input,float* output) {
  __shared__ float tile[256];
  int i=blockIdx.x*256+threadIdx.x;
  unsigned int dst=static_cast<unsigned int>(__cvta_generic_to_shared(&tile[threadIdx.x]));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(dst),"l"(input+i) : "memory");
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  output[i]=tile[threadIdx.x];
}

__global__ void bf16_wmma(const __nv_bfloat16* a,const __nv_bfloat16* b,float* c) {
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
  wmma::fragment<wmma::accumulator,16,16,16,float> cf;
  wmma::fill_fragment(cf,0.f);
  wmma::load_matrix_sync(af,a,16);
  wmma::load_matrix_sync(bf,b,16);
  wmma::mma_sync(cf,af,bf,cf);
  wmma::store_matrix_sync(c,cf,16,wmma::mem_row_major);
}

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
  constexpr int n=65536;
  std::vector<float> input(n),output(n),expected(256),actual(256);
  std::vector<__nv_bfloat16> ah(256),bh(256);
  for(int i=0;i<n;++i) input[i]=float(i%113-56)*0.125f;
  for(int row=0;row<16;++row) for(int col=0;col<16;++col) {
    ah[row*16+col]=__float2bfloat16(float((row+col)%9-4)*0.25f);
    bh[col*16+row]=__float2bfloat16(float((row*3+col)%7-3)*0.25f);
  }
  for(int row=0;row<16;++row) for(int col=0;col<16;++col) {
    float sum=0; for(int k=0;k<16;++k) sum+=__bfloat162float(ah[row*16+k])*__bfloat162float(bh[col*16+k]); expected[row*16+col]=sum;
  }
  float *di=nullptr,*dout=nullptr,*dc=nullptr; __nv_bfloat16 *da=nullptr,*db=nullptr;
  CUDA_CHECK(cudaMalloc(&di,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&dout,n*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&da,256*sizeof(__nv_bfloat16))); CUDA_CHECK(cudaMalloc(&db,256*sizeof(__nv_bfloat16))); CUDA_CHECK(cudaMalloc(&dc,256*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(di,input.data(),n*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(da,ah.data(),256*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db,bh.data(),256*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
  async_copy<<<n/256,256>>>(di,dout); CUDA_CHECK(cudaGetLastError());
  bf16_wmma<<<1,32>>>(da,db,dc); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(),dout,n*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual.data(),dc,256*sizeof(float),cudaMemcpyDeviceToHost));
  bool copy_ok=input==output,mma_ok=true; float max_error=0;
  for(int i=0;i<256;++i) { float err=std::fabs(actual[i]-expected[i]); if(!std::isfinite(actual[i])||err>1e-5f) mma_ok=false; if(err>max_error) max_error=err; }
  std::printf("{\"device\":\"%s\",\"compute_capability\":[%d,%d],\"cp_async_4byte_copy\":{\"elements\":%d,\"correct\":%s},\"wmma_bf16_fp32\":{\"shape\":[16,16,16],\"correct\":%s,\"absolute_tolerance\":1e-5,\"max_absolute_error\":%.9g},\"performance_claim\":null}\n",p.name,p.major,p.minor,n,copy_ok?"true":"false",mma_ok?"true":"false",max_error);
  CUDA_CHECK(cudaFree(di)); CUDA_CHECK(cudaFree(dout)); CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
  return copy_ok&&mma_ok?0:4;
}
