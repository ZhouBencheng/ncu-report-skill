#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA error: %s (%s:%d)\n", cudaGetErrorString(e),__FILE__,__LINE__); return 2; } } while(0)

__global__ void vector_copy(const float* a, float* b, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) b[i]=a[i];
}
__global__ void block_reduce(const float* a, float* out, int n) {
  __shared__ float values[256];
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  values[threadIdx.x]=i<n?a[i]:0.f;
  __syncthreads();
  for(int d=128;d;d/=2) {
    if(threadIdx.x<d) values[threadIdx.x]+=values[threadIdx.x+d];
    __syncthreads();
  }
  if(threadIdx.x==0) out[blockIdx.x]=values[0];
}
__global__ void matrix_multiply(const float* a,const float* b,float* c,int n) {
  int row=blockIdx.y*blockDim.y+threadIdx.y;
  int col=blockIdx.x*blockDim.x+threadIdx.x;
  if(row<n&&col<n) { float x=0; for(int k=0;k<n;++k) x+=a[row*n+k]*b[k*n+col]; c[row*n+col]=x; }
}

int main(int argc,char** argv) {
  int count=0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if(count<1) return 3;
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p,0));
  int runtime=0,driver=0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  std::printf("{\"device\":\"%s\",\"cc_major\":%d,\"cc_minor\":%d,\"sm_count\":%d,\"total_global_memory_bytes\":%zu,\"shared_memory_per_block_bytes\":%zu,\"shared_memory_per_block_optin_bytes\":%zu,\"shared_memory_per_sm_bytes\":%zu,\"l2_bytes\":%d,\"registers_per_sm\":%d,\"warp_size\":%d,\"max_threads_per_sm\":%d,\"max_threads_per_block\":%d,\"max_blocks_per_sm\":%d,\"integrated\":%s,\"runtime_version\":%d,\"driver_api_version\":%d}\n",p.name,p.major,p.minor,p.multiProcessorCount,p.totalGlobalMem,p.sharedMemPerBlock,p.sharedMemPerBlockOptin,p.sharedMemPerMultiprocessor,p.l2CacheSize,p.regsPerMultiprocessor,p.warpSize,p.maxThreadsPerMultiProcessor,p.maxThreadsPerBlock,p.maxBlocksPerMultiProcessor,p.integrated?"true":"false",runtime,driver);
  if(argc>1&&std::strcmp(argv[1],"--info")==0) return 0;
  constexpr int n=262144,blocks=(n+255)/256,m=64;
  std::vector<float> input(n),copy(n),partial(blocks),ma(m*m),mb(m*m),mc(m*m);
  for(int i=0;i<n;++i) input[i]=float(i%17-8);
  for(int i=0;i<m*m;++i) { ma[i]=float(i%9-4)*0.25f; mb[i]=float(i%7-3)*0.25f; }
  float *a=nullptr,*b=nullptr,*r=nullptr,*da=nullptr,*db=nullptr,*dc=nullptr;
  CUDA_CHECK(cudaMalloc(&a,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&b,n*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&r,blocks*sizeof(float))); CUDA_CHECK(cudaMalloc(&da,m*m*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db,m*m*sizeof(float))); CUDA_CHECK(cudaMalloc(&dc,m*m*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(a,input.data(),n*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(da,ma.data(),m*m*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db,mb.data(),m*m*sizeof(float),cudaMemcpyHostToDevice));
  vector_copy<<<blocks,256>>>(a,b,n); CUDA_CHECK(cudaGetLastError());
  block_reduce<<<blocks,256>>>(a,r,n); CUDA_CHECK(cudaGetLastError());
  matrix_multiply<<<dim3(4,4),dim3(16,16)>>>(da,db,dc,m); CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(copy.data(),b,n*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(partial.data(),r,blocks*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mc.data(),dc,m*m*sizeof(float),cudaMemcpyDeviceToHost));
  bool copy_ok=copy==input,reduce_ok=true,matmul_ok=true; float max_error=0;
  for(int bidx=0;bidx<blocks;++bidx) { float expect=0; for(int i=bidx*256;i<(bidx+1)*256&&i<n;++i) expect+=input[i]; if(partial[bidx]!=expect) reduce_ok=false; }
  for(int row=0;row<m;++row) for(int col=0;col<m;++col) { float expect=0; for(int k=0;k<m;++k) expect+=ma[row*m+k]*mb[k*m+col]; float err=std::fabs(mc[row*m+col]-expect); if(!std::isfinite(mc[row*m+col])||err>1e-5f) matmul_ok=false; if(err>max_error) max_error=err; }
  std::printf("{\"vector_copy\":{\"elements\":%d,\"correct\":%s},\"block_reduce\":{\"elements\":%d,\"correct\":%s},\"matrix_multiply\":{\"shape\":[64,64,64],\"dtype\":\"float32\",\"correct\":%s,\"max_absolute_error\":%.9g},\"performance_claim\":null}\n",n,copy_ok?"true":"false",n,reduce_ok?"true":"false",matmul_ok?"true":"false",max_error);
  CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(r)); CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
  return copy_ok&&reduce_ok&&matmul_ok?0:4;
}
