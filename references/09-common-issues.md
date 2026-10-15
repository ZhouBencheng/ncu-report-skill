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
   cuobjdump --dump-function-names ./my_harness
   ```
   Match against the visible string.
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

## PM sampling returns nothing

1. **You didn't request it.** Add `--section PmSampling --section PmSampling_WarpStates` to the ncu invocation.
2. **vGPU / MIG environment.** PM sampling isn't supported under virtualization. Use metric-based (non-timeline) analysis only.
3. **Kernel too short.** Kernels under ~20 µs produce few PM samples; what comes back is dominated by warmup noise.
4. **Specific PM metric just isn't available on your GPU / driver / ncu combination.** Some `pmsampling:sm__throughput.*` or `pmsampling:dram__throughput.*` variants may return empty instance arrays even when other `pmsampling:smsp__warps_issue_stalled_*` series work fine on the same report. Always check `m.num_instances() > 0`. An empty SM/DRAM timeline is an **absent measurement, not zero traffic** — it is *not* a reliable stall proxy and does not substitute for the SM/DRAM signal. The stall-reason timelines answer a different question (which stall reasons dominate over time); report the SM/DRAM series as unavailable rather than inferring throughput or a stall conclusion from its absence.

---

## ncu takes forever to finish

1. **`--set full` needs 45+ replay passes.** That's normal — each pass reruns the kernel for a different metric group. If your kernel takes 3 ms, full profile takes ~15 s; 3 ms → 300 ms kernels are miserable. Mitigation: profile a smaller representative workload if possible.
2. **Kernel launches aren't isolated.** If your binary does other expensive work (data loading, CUDA context init) before every kernel launch, that runs on every replay too. Move it outside the profile window (ncu only profiles kernel launches matching `-k`).
3. **Don't use `-G` (debug).** It regresses performance ~100× and is useless for perf profiling.

---

## Kernel crashes / produces NaN only under ncu

1. **Profiler clock jitter.** Between replays, ncu resets GPU state. Kernels that depend on specific uninitialized values (bad practice) can behave differently. Fix: initialize all inputs.
2. **Out-of-order pre-replay memory**: ncu saves/restores GPU memory between replays but that can expose latent bugs like reading uninitialized global memory.

---

## Metric returns `None`

1. **Wrong metric name.** See `08-b200-metric-names.md`. Many stock docs use names that don't exist on B200.
2. **Metric not in the collected sections.** Add the relevant `--section` or `--set`.
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

```bash
find /usr/local/cuda* -name "ncu_report*" -type f 2>/dev/null
# e.g. /opt/nvidia/nsight-compute/2026.1.1/extras/python/ncu_report.py
export PYTHONPATH=$PYTHONPATH:/opt/nvidia/nsight-compute/2026.1.1/extras/python
python3 -c "import ncu_report; print('OK')"
```

If there's still an `ImportError`, check that the module is compatible with your Python version. The `_ncu_report*.so` compiled extension alongside `ncu_report.py` is built for one specific Python version.

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

`-k "regex:..."` must match the demangled name. TVM-FFI wraps kernels in `__global__ void kernel(...)` with a fixed name — check with `cuobjdump --dump-function-names <path to .so>`.

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

1. **Lock GPU clocks:**
   ```bash
   sudo nvidia-smi -lgc <boost_clock_mhz>    # check with nvidia-smi -q -d CLOCK
   # profile
   sudo nvidia-smi -rgc                      # unlock
   ```
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

### Thor: an occupancy-labelled counter exceeds 100%

In the 2026-10-13 original-FP32 ActiveVLN gate/SiLU/multiply experiment, the
40-register native kernel reported
`sm__warps_active.avg.pct_of_peak_sustained_active=110.96731945718507%`.
Keep the original typed value, unit and report identity, flag the anomaly, and
exclude it from physical achieved-occupancy conclusions. Do not clamp to 100 or
guess a denominator, clock or replay cause. This is not a blanket rule that all
`pct_of_peak` metrics above 100 are invalid; their definitions differ.

The same kernel's long-scoreboard count was **96.3366% of 6,988 ordinary PC
samples**, not 96.3366% of elapsed time. Its mapped FFMA locations are consumers,
not proof of a unique load producer. DRAM read/write/throughput metrics were
unavailable across both reports: preserve missingness, never fill it with zero
or conclude bandwidth saturation from those absent measurements.

The complete FFN used five kernels before and three after. Sixteen consecutive
eager calls with the same weights showed **1.64% lower per-call latency**
(B/C 1.01664065, interval [1.01354657, 1.02029359]); this did not predict a stable
gain in the actual 36-layer model. The separate initial-request E2E primary
ratio was 1.00028814, interval [0.99515433, 1.00389403], with one slower process
pair. That predefined gate failed and the candidate was not integrated. Keep
ordinary timings separate from NCU replay duration, and keep same-weight
microbenchmarks separate from full-model evidence. These intervals describe the
fixed experiments, not a population or general navigation-workload guarantee.

A later guard-only revision kept the same native kernel bytes and reduced
eligibility-check CPU cost (B/C 1.6948), but its separate full-model E2E primary
ratio was 1.00027059, interval [0.99740037, 1.00300278]. Both process-pair point
estimates were positive; the predefined interval gate still failed. The bounded
experiment ended without integration or further samples selected by the result.
Neither fewer kernels nor faster CPU guards establish an end-to-end benefit.

Evidence is external to the skill: campaign root
`/home/cilab/zbc/embodiinfer-operators/profile/embodiinfer-thor-sm110-loop-20261013`
(local mirror `/Users/zbc/workspace/kda/profile/embodiinfer-thor-sm110-loop-20261013`).
Verify the following relative paths and SHA256 values before reusing the claims;
unavailable receipts remain unavailable evidence:

- `analysis/active-gate-v2-performance-independent-review.md`:
  `fa9365aed499f3ab56bba1ed9d764ebea289537c23cedcb19f15d38be7c239d5`.
- `analysis/active-gate-v2-ncu-independent-evidence.json`:
  `1003cc3592ca5ea8f10a85f506b3d85149d11f7ee9ec88ee6949d1d848530231`.
- `analysis/active-gate-v2-e2e-actual-independent-review.md`:
  `16ea820b4f560f186f61bacfdee2e0f37c7a4eb26d9f7ffdfffba6ac2b84cdb2`.
- `analysis/active-gate-v2-e2e-actual-independent-evidence.json`:
  `89dd19b26416af81b66d6d7988b2d8e45388fe4007298a23644b2f5f7f8bfa61`.
- `analysis/active-gate-v3-guard-actual-independent-evidence.json`:
  `0a2d48271115bed2affa8fc0a260329c82b7086c3313298998579e62efde3f63`.
- `analysis/active-gate-v3-e2e-actual-independent-evidence.json`:
  `6361dc4a1ca615e583b26a17beb15173b234bbe49b85edec295698da76a339ed`.

### Orin: a half-opcode spelling is not a precision diagnosis

In one native SM87 ActiveVLN down-projection artifact, `HFMA2.MMA` at SASS
addresses `0x180` and `0x520` constructs constants from RZ/immediates; neither
instruction consumes model data. The loaded FP32 inputs instead flow through
FFMA chains and FADD/SHFL/shared-memory reduction, consistent with the retained
PTX `fma.rn.f32` / `add.rn.f32` data path. Determine precision by following the
actual input-to-output register definitions and uses in the exact compiled
artifact, not by counting an opcode name or treating this example as a whitelist.

This was a source/binary review only: it permitted the fixed correctness test,
and did not establish numerical correctness, speedup, NCU mechanism, model
equivalence, integration or BF16 qualification. Evidence is external under
`/Users/zbc/workspace/kda/profile/orin-m1-ffn-20261014`:

- `analysis/candidate-active-down-v1-independent-review.md`:
  `df99b81b173c939445566ccf86c326683e6d1c778a974143bfb7b4bb086a242c`.
- `analysis/candidate-active-down-v1-independent-review.json`:
  `cd03e021c8c9d20eb6358e016a6b3ffb35be06807a051bd241f3ce0695c7ffa7`.

The receipt binds manifest
`ae7c4016280ea9da46f272265cf318e74915b6402a0cea1bd2c845fd35a62476`
and library `5ce533f6f22f4d1f47b239c54e11fcc427b58dde2cb7189bca82ca9963d25f2b`.
If those artifacts or receipts are unavailable, retain that evidence limitation.

### "`sm__throughput = X%`, is that good?"

It depends on the kernel type:
- GEMM / matmul: should be 50%+ on B200. Below 30% is bad.
- Element-wise / reduction: usually 10-30%, because they're DRAM-BW-bound.
- Attention / recurrence kernels: varies wildly; compare against a reference implementation.

Always check Speed-of-Light alongside: `dram__bytes_read.sum.pct_of_peak_sustained_elapsed`. If DRAM is saturated, low SM throughput is expected and OK. If DRAM is idle AND SM is idle, you're latency-bound.

### "The details page says `Est. Speedup: X%` — is that reliable?"

Yes, mostly. NCU's rule engine does a reasonable job estimating individual rule impact. Caveats:

- The sum of all `Est. Speedup`s is usually > 100%, because rules overlap (fixing A might also help B). Don't add them.
- Rules are per-pattern; the rule engine doesn't know which one is hardest/easiest to fix in your codebase.
- Use `Est. Speedup: X%` to rank patterns by magnitude; use your judgement for ease of implementation.
