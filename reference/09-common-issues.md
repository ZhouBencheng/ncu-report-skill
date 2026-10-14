# Common Issues & Gotchas

Collected solutions for the recurring frustrations of profiling CUDA kernels.

---

## ncu permissions

### `ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU Performance Counters on the target device`

Two solutions:

**A) Use sudo (simplest on dedicated servers):**
```bash
sudo ncu [...]
```

**B) Make it persistent (preferred on shared servers):**
```bash
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/ncu.conf'
sudo update-initramfs -u
# reboot, then regular user can run ncu
```

**C) In a container without `sudo`:** the container needs counter access from whoever starts it (the host driver
setting above, or the container started with `--cap-add SYS_ADMIN`). Report the error to them instead of working
around it.

## ncu warnings

### `Unable to access the following 6 metrics: ctc__rx_bytes_data_user.sum, ...`

Harmless. The `ctc__*` metrics count chip-to-chip (C2C) link traffic; a system without that link cannot collect them, and `--set full` asks for them anyway.

### `No source files were imported` / `Failed to import the following source files`

`--import-source yes` reads the source where ncu runs, at collection time, from the paths the build recorded (`ncu --import ... --page source` shows them). If those paths do not exist there — a prebuilt library, a build directory that is gone, a GPU job in another container — fetch the matching source somewhere ncu can see and pass `--source-folders` (searched recursively), then collect again: the source cannot be added to an existing report. A kernel without line info has no source to import.

### `Could not deploy stock section files to "/home/USER/Documents/NVIDIA Nsight Compute/..."`

Set `HOME` to a writable directory:
```bash
export HOME=/any/writable/path
ncu [...]
```

This warning is harmless but noisy. ncu falls back to reading from the CUDA install dir.

---

## `-k "regex:..."` matches nothing

1. **Use the demangled name.** Templates produce something like `void my_kernel<(int)8, (int)256>(...)`. Check:
   ```bash
   cuobjdump -res-usage ./my_harness | grep Function | cu++filt
   ```
   Match against the visible string, or run the survey (`03-collection.md`, Recipe 0), which lists each launch's demangled name.
2. **Escape regex metacharacters.** `<` and `>` don't need escaping in most regex flavors, but be careful with parentheses.
3. **The kernel might not have been launched.** Run the harness without ncu and confirm the kernel actually runs.

---

## Source view is empty / `action.source_info(pc)` returns None

The binary was compiled without `-lineinfo`. Add it to the nvcc invocation:
```bash
nvcc -O2 -std=c++17 -lineinfo -gencode=... kernel.cu -o harness
```

For JIT / framework-integrated builds:

- **TVM-FFI**: hard to inject `-lineinfo`. Easiest fix: build a standalone harness (see `02-harness-guide.md`).
- **PyTorch `torch.utils.cpp_extension.load`**: pass `extra_cuda_cflags=["-lineinfo"]`. But also switch out of `-O3 -G` to avoid heavy debug instrumentation.
- **CUTLASS**: pass `-lineinfo` via `CMAKE_CUDA_FLAGS`.
- **Triton**: harder — Triton's JIT codegen ignores user nvcc flags. To do source-level on Triton, dump the generated PTX, rebuild as a standalone, and profile.

---

## Source view shows only CUTLASS / CuTe / library headers

ncu maps each PC to its **innermost** inlined frame. In template-heavy code nearly every instruction is inlined from a library header (`mma_sm80.hpp` for every HMMA, `barrier.h` for every mbarrier wait), so the hottest "lines" are the same headers for every kernel. Pass the profiled binary (executable, `.so` or cubin) to `extract_stall_hotspots.py --binary`: it reads the full inline chain with `nvdisasm -gi` and ranks the lines in the kernel function itself. It reports how many sampled PCs' SASS matched the report; a mismatch means the binary is not the one that was profiled.

---

## PM sampling returns nothing

1. **You didn't request it.** `--set full` includes the PmSampling section; add `--section PmSampling_WarpStates` for the stall-reason timelines.
2. **vGPU / MIG environment.** PM sampling isn't supported under virtualization. Use metric-based (non-timeline) analysis only.
3. **Kernel too short.** Kernels under ~20 µs produce few PM samples; what comes back is dominated by warmup noise.
4. **You asked for a name the report does not have.** ncu 2026.1 names the PmSampling series `<UNIT>.<group>.<metric>` (e.g. `TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed`); `pmsampling:sm__throughput.*` and `pmsampling:dram__throughput.*` from older docs do not exist. `plot_timeline.py --list` prints what the report has.

---

## ncu takes forever to finish

1. **`--set full` needs dozens of replay passes.** That's normal — each pass reruns the kernel for a different metric group. If your kernel takes 3 ms, full profile takes ~15 s; 3 ms → 300 ms kernels are miserable. Mitigation: profile a smaller representative workload if possible.
2. **Kernel launches aren't isolated.** If your binary does other expensive work (data loading, CUDA context init) before every kernel launch, that runs on every replay too. Move it outside the profile window (ncu only profiles kernel launches matching `-k`).
3. **Don't use `-G` (debug).** It regresses performance ~100× and is useless for perf profiling.

---

## Kernel crashes / produces NaN only under ncu

1. **Profiler clock jitter.** Between replays, ncu resets GPU state. Kernels that depend on specific uninitialized values (bad practice) can behave differently. Fix: initialize all inputs.
2. **Out-of-order pre-replay memory**: ncu saves/restores GPU memory between replays but that can expose latent bugs like reading uninitialized global memory.

---

## Metric returns `None`

1. **Wrong metric name.** See `08-metric-names.md`. Many stock docs use names `--set full` does not collect on Blackwell.
2. **Metric not in the collected sections.** Add the relevant `--section` or `--set`, or ask for it with `--metrics` (some metrics are in no section).
3. **Value is legitimately missing.** Some metrics (like tensor pipe counters) return 0 rather than None when the feature wasn't used; but others return None for hardware not present.

Always wrap metric reads in a helper that returns a default:
```python
def safe(action, name, default=None):
    try:
        return action[name].value()
    except Exception:
        return default
```

---

## `ncu_report` import fails

`ncu_report.py` ships in `<Nsight Compute install>/extras/python`. The helpers find the install from the `ncu` on PATH, as the CUDA toolkit's `ncu` launcher does (`/opt/nvidia/nsight-compute/<version>` for .deb/.rpm installs, `<toolkit>/nsight-compute-<version>` for .run installs; the newest wins):

```bash
python3 <skill>/helpers/ncu_utils.py      # prints where it found ncu_report, or why not
```

If `ncu` is not on PATH or the install is elsewhere, put its `extras/python` on `PYTHONPATH` yourself. If there's still an `ImportError`, check that the module is compatible with your Python version.

---

## TVM-FFI specific

### "I can't profile my kernel, it's built by `tvm_ffi.cpp.build`"

- Locate the cached `.so` and `kernel.cu`:
  ```
  ~/.cache/flashinfer_bench/cache/tvm_ffi/<solution_hash>/
  ```
- You'll see `build.ninja` with the nvcc invocation — note it does *not* include `-lineinfo`.
- **Workaround:** build a standalone harness from the cached `kernel.cu` (see `02-harness-guide.md`).
- **Alternative:** inject `-lineinfo` into `build.ninja` and recompile manually. But this breaks on the next TVM rebuild.

### "The Python benchmarking script runs but ncu sees no matching kernel"

`-k "regex:..."` must match the demangled name. TVM-FFI wraps kernels in `__global__ void kernel(...)` with a fixed name — list them with `cuobjdump -res-usage <path to .so> | grep Function` (mangled; `cu++filt` demangles), or run the survey in `03-collection.md`.

---

## PyTorch specific

### Profiling a PyTorch model's kernel

1. Identify the kernel. Use `torch.profiler` to name it, or look for the generated kernel from `torch.compile`.
2. The kernel is often auto-generated (Triton, CUTLASS, cuDNN). For Triton kernels specifically, Triton recompiles between runs and the kernel name changes. Profile a single script invocation.
3. For `torch.compile`-generated Triton, inspect `TORCH_LOGS=+dynamo` or `TORCH_COMPILE_DEBUG=1` to see the emitted code.

### Profiling CUDA Graph-captured kernels

ncu handles CUDA Graph launches fine — each captured kernel shows up as a separate "kernel launch". Use `-k` regex + `-c N` to target.

---

## Reproducibility

### Results jitter between runs

1. **Check the clock.** ncu locks the clock by default (`boost` in ncu 2026.1, `base` before); a run with
   `--clock-control none` and no outer lock profiles at whatever clock the GPU picks. Compare
   `sm__cycles_elapsed.avg.per_second` between the reports, and lock your own clock only as
   [`03-collection.md`](03-collection.md#gpu-clocks) describes.
2. **Enable persistent mode** (avoids driver unload between invocations):
   ```bash
   sudo nvidia-smi -pm 1
   ```
3. **Pin the CUDA stream** explicitly rather than relying on default stream.

### Reports don't match colleague's results

- Check ncu version (`ncu --version`). Metric names change between major versions.
- Check GPU driver version (`nvidia-smi`). Some metrics only exist on certain drivers.
- Check exact nvcc invocation — a stray `-G` or missing `-lineinfo` makes a big difference.

---

## Output interpretation

### "`sm__throughput = X%`, is that good?"

It depends on the kernel type:
- GEMM / matmul: should be 50%+ on B200 / B300. Below 30% is bad.
- Element-wise / reduction: usually 10-30%, because they're DRAM-BW-bound.
- Attention / recurrence kernels: varies wildly; compare against a reference implementation.

Always check Speed-of-Light alongside: `dram__bytes_read.sum.pct_of_peak_sustained_elapsed`. If DRAM is saturated, low SM throughput is expected and OK. If DRAM is idle AND SM is idle, you're latency-bound.

### "The details page says `Est. Speedup: X%` — is that reliable?"

It is a good pointer to where to look, and often in the right ballpark — but check each estimate against the counters it rests on before ranking by it. Caveats:

- The sum of all `Est. Speedup`s is usually > 100%, because rules overlap (fixing A might also help B). Don't add them.
- Rules are per-pattern; the rule engine doesn't know which one is hardest/easiest to fix in your codebase.
- Some estimates can be far off: a "small grid" estimate assumes the work spreads perfectly over more CTAs; a shared-memory bank-conflict estimate can rest on inconsistent wavefront counters; an L2-compression estimate applies only to data that could be compressed; an "uncoalesced loads" finding can be a scalar broadcast load. Confirm with the source counters (`stall_hotspots_<tag>.txt`) and the raw metrics.
- Use `Est. Speedup: X%` to rank patterns by magnitude; use your judgement for ease of implementation.
