# Diagnosis Playbook — Pattern → Cause → Fix

For each observed NCU signal, what does it typically mean, and what's the first fix to try? This synthesizes the Blackwell programming principles (the companion `blackwell-cuda-programming.md` at the repo root) with the profiling signals.

Read this after you've gathered the metrics (via [`05-analysis-dimensions.md`](05-analysis-dimensions.md)) — here you translate metrics into diagnoses and fix directions.

---

## How to use this doc

For each *observation* below, read:

- **Signals** — what specific metric values flag this pattern.
- **Why** — the underlying cause.
- **First-line fix** — the cheapest change to try.
- **Deeper fixes** — when first-line isn't enough.
- **Exceptions** — kernel types where this pattern is actually *expected* and should be left alone.

Most kernels will match 2-4 patterns simultaneously. **Rank them by magnitude** using NCU's `Est. Speedup: X%` fields (from `--page details` or `rules_<tag>.txt`) and the stall-percentage breakdown, after checking each estimate against the counters it rests on. Fix the biggest one first.

---

## Pattern A — Small grid / SM idle

**Signals:**
- `launch__waves_per_multiprocessor < 0.5`
- `launch__grid_size < device__attribute_multiprocessor_count` (e.g., 64 blocks on a 148-SM B200 / B300)
- NCU rule: *"The grid for this launch is configured to execute only N blocks, which is less than the M multiprocessors used."* with `Est. Speedup: 50-90%`

**Why:** each CTA occupies at most one SM; with fewer CTAs than SMs, some SMs are completely idle throughout the kernel.

**First-line fix:** increase grid size. Look for a dimension the kernel currently doesn't parallelize:
- Add a split along `K` (split-K for reductions / attention).
- Split across heads / channels if grouped.
- Use Grid-stride loops so one block does multiple work units — but only if work units are cheap.

**Deeper fixes:**
- **Persistent kernel**: launch one block per SM, each block dequeues work items from an atomic counter. Good for dynamic-shape cases.
- **Fuse with adjacent kernels** so more work fits in one launch.

**Exceptions:**
- LLM decode (batch=1, query_len=1) is fundamentally small. Split-K over KV length is the standard mitigation.
- Final reduction stages of a multi-level reduction are naturally small; fuse them into the producing kernel.

**Cross-ref:** Blackwell principle 1 (the companion `blackwell-cuda-programming.md` at the repo root).

---

## Pattern B — Tail effect (variable-length inputs)

**Signals:**
- Multi-workload: `max_seq_len / avg_seq_len > 3` in input distribution.
- Per-SM active cycles span 5-100× between slowest and fastest SM (from `--page details` distribution) — or much less when the longest CTAs start late: then the SM-activity timeline's falling edge is the signal.
- PM timeline shape: long gradual tail at the end (visible via `plot_timeline.py`).
- `launch__waves_per_multiprocessor > 1.05` with partial last wave.
- NCU rule: `"partial wave may account for up to X% of the total runtime"`.

**Why:** each CTA iterates some variable-size inner loop. When sequences have vastly different lengths, a few long-sequence CTAs keep running after everyone else finished.

**First-line fix (cheap):**
- **Packed batching / sorting**: sort inputs by length (at the application level) so CTAs running concurrently do roughly equal work.
- **Split long sequences across more CTAs**: add a `split_factor` grid dimension; each CTA handles `ceil(seq_len / split_factor)` tokens, and a small post-reduction combines partials.

**Deeper fixes:**
- **Chunkwise kernel**: break each sequence into fixed-size chunks, process chunks in parallel, then stitch with a small recurrence. This is the approach of flash-linear-attention's `chunk_delta_rule_fwd` for Mamba/GLA-style recurrences.
- **Classify-and-dispatch**: short sequences go through the simple path (one CTA per seq), long sequences through the chunked path.

**Exceptions:**
- Short kernels (< 10 µs) where partial-wave cost is absolute-small.
- Workloads where you already pre-sort / pre-pack.

**Cross-ref:** Blackwell principle 11.

---

## Pattern C — Uncoalesced global loads

**Signals:**
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum > 5` (ideal is 4).
- NCU rule: *"uncoalesced global accesses resulting in N excessive sectors (X% of the total)"*.
- NCU rule: *"On average, only Y of the 32 bytes transmitted per sector are utilized"*.
- Primary stall reason on the offending load line is `long_scoreboard`.

**Why:** lanes in a warp access non-contiguous addresses; hardware fetches extra sectors that only a few lanes use.

**First-line fix:** rework the thread ↔ data mapping:
- If current pattern is `x[lane * K + i]` (stride K), flip to `x[lane + i * 32]` (coalesced).
- Check AoS layouts: `struct { float a, b; } arr[N]` → `struct { float a[N], b[N]; }` so each field is a separate coalesced stream.

**Deeper fixes:**
- Use shared memory as a transposer: coalesced-load to shared, then arbitrary-access from shared.
- Vectorize: replace scalar `LDG.E` with `LDG.E.64` / `LDG.E.128` (use `float2` / `float4` / `ushort2` types).

**Exceptions:**
- Gather/scatter by random index (sparse matmul, embedding lookup) — fundamentally uncoalesced. Sort the indices for locality if possible.
- Graph / tree traversal.

**Cross-ref:** Blackwell principles 2, 13.

---

## Pattern D — Sparse writes (low store efficiency)

**Signals:**
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio < 16` (ideal is 32).
- `l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct` lower than expected.
- Code contains patterns like `if (lane_id < K) { output[...] = ... }`.

**Why:** only a subset of warp lanes write, so the L1 store buffer flushes half-empty sectors.

**First-line fix:** pack the write. Have the warp collectively produce `K` values first (via shuffle or shared memory reduction), then have exactly `K` contiguous lanes perform `K` consecutive writes.

If `K ≥ 32`: all lanes can write; make sure the per-lane index is contiguous.

If `K < 8`: consider batching multiple iterations' results into a vectorized write (e.g., 4 iterations' output packed into a single `float4`).

**Deeper fixes:**
- Write into shared memory first, then do a coalesced global store at the end of the block.

**Exceptions:**
- Histogram / scatter (inherently sparse) — different optimization path, see Pattern G.

---

## Pattern E — Latency-bound (long-scoreboard-dominated)

**Signals:**
- `smsp__pcsamp_warps_issue_stalled_long_scoreboard / smsp__pcsamp_sample_count > 0.40`.
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio > 3`.
- `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed < 10%` (→ not DRAM-bandwidth-bound).
- Hotspot lines are global loads (check `stall_hotspots_<tag>.txt`).

**Why:** warps issue a load, then stall waiting for it to return before the next dependent op. Usually combined with low occupancy or insufficient ILP.

**First-line fix:** increase in-flight memory requests:
- **Unroll the load loop** so 4-8 loads are issued before any value is used. Compiler + hardware reorders.
- **Add more independent warps** — raise occupancy (Pattern J).
- **`cp.async` (Ampere+) / TMA (Hopper+) / tcgen05.cp (Blackwell)** for bulk async loads that don't block issue.

**Deeper fixes:**
- Software pipelining: while tile N is being computed, pre-load tile N+1 into shared memory.
- Move reused data to shared memory so subsequent loads hit L1.

**Exceptions:**
- Pointer chasing / graph traversal — data dep chain is fundamental.

**Cross-ref:** Blackwell principles 7, 15.

---

## Pattern F — Compute-bound but not on tensor cores

**Signals:**
- `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active > 50%`.
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed = 0%`.
- Workload is matmul-ish (GEMM, attention, conv).

**Why:** kernel uses scalar FMA via the ALU pipe instead of tensor cores. On B200, tensor cores can do 16× the FMA throughput of scalar pipes for BF16→FP32.

**First-line fix:** use `WMMA` (Ampere+) / `wgmma` (Hopper) / `tcgen05.mma` (Blackwell). If hand-rolling is too much, use CUTLASS 4.x or cuBLAS, which are already tuned for the target arch.

**Deeper fixes:**
- Restructure data layout to meet MMA tile-shape constraints (e.g., `m16n8k16` for BF16).
- Use shared memory + TMA (Hopper) / TMEM (Blackwell) staging.

**Exceptions:**
- Non-matrix workloads (reduction, sort, element-wise) — tensor cores don't help.
- Small matrices (M, N, K < 32) — tensor-core tiles are too coarse.

**Cross-ref:** Blackwell principle 10; the Blackwell doc's section on tcgen05 has PTX examples.

---

## Pattern G — Atomics contention

**Signals:**
- `long_scoreboard` samples concentrate on `ATOM` / `RED` SASS instructions.
- `lts__t_sectors_op_atom.sum` or `lts__t_sectors_op_red.sum` is large.
- L2 throughput is high but compute throughput is low.

**Why:** many threads atomically updating few locations → serialization.

**First-line fix:** hierarchical reduction.
- Within-warp: `__shfl_down_sync` (no atomic).
- Within-block: shared memory reduction (no atomic).
- Between blocks: single atomic at the end per block.

**Deeper fixes:**
- Shared-memory histogram that flushes to global in one coalesced pass.
- Bucketing: thread writes to `output[tid % N_buckets]`, followed by a merge kernel.

**Exceptions:**
- NCCL-style communication — atomics are fundamental there.

**Cross-ref:** Blackwell principle 12.

---

## Pattern H — Shared-memory bank conflicts

**Signals:**
- `l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed` high for shared-mem ops.
- `short_scoreboard` stalls concentrated on shared-memory load lines.
- Access pattern has regular strides that align to bank boundaries.
- Confirm per instruction before trusting NCU's bank-conflict rule: the per-PC metrics `memory_l1_wavefronts_shared` and `memory_l1_wavefronts_shared_ideal` (and `derived__memory_l1_wavefronts_shared_excessive`) of a `--set full` report give each shared-memory instruction's actual and conflict-free wavefronts. Summed excess well below the rule's estimate means the estimate is off.

**Why:** shared memory has 32 banks; same-bank accesses serialize.

**First-line fix:** padding. `__shared__ float tile[32][33]` instead of `[32][32]` breaks regular bank alignment.

**Deeper fixes:**
- Swizzle: XOR-scramble indices so accesses spread across banks.
- Restructure data layout so warp lanes access different banks.

**Exceptions:**
- Broadcast reads (all lanes read same address) are conflict-free.
- Low shared-mem access volume — don't bother.

**Cross-ref:** Blackwell principle 4.

---

## Pattern I — Synchronization overhead

**Signals:**
- `smsp__pcsamp_warps_issue_stalled_barrier` > 20% of samples.
- Source hotspot line is `BAR.SYNC`.

**Why:** `__syncthreads()` waits for the slowest warp. Combined with any per-warp work imbalance, this amplifies.

**First-line fix:**
- Replace block-level syncs with warp-level primitives (`__shfl_sync`, `__ballot_sync`, `__syncwarp`) where only warp-scoped synchronization is needed.
- Reduce total sync count — consolidate multiple synchronized phases.

**Deeper fixes:**
- Warp-specialized execution: producer warps and consumer warps with mbarrier instead of `__syncthreads`.

**Cross-ref:** Blackwell principle 16.

---

## Pattern J — Low achieved vs theoretical occupancy

**Signals:**
- `sm__maximum_warps_per_active_cycle_pct > 50` but `sm__warps_active.avg.pct_of_peak_sustained_active << 50`.
- NCU rule: *"The difference between calculated theoretical (X%) and measured achieved occupancy (Y%) ..."*.

**Why:** Theoretical occupancy is the max warps that *could* be resident. Achieved is how many are *actually* running. Gap is caused by: stalls (leaves slots empty), imbalance (some SMs empty), short kernel (warmup dominates).

**Reading:** if the gap is large AND Pattern B (tail effect) is present, fixing imbalance will close the gap. If no imbalance, look at stall reasons (Pattern E, H, I).

**First-line fix:** look for the stall reason causing the gap and address that pattern.

---

## Pattern K — Register spill

**Signals:**
- `smsp__sass_inst_executed_op_local_ld.sum > 0` or `smsp__sass_inst_executed_op_local_st.sum > 0`.
- NCU rule: *"N bytes spilled to local memory"* in Instruction Statistics.
- `launch__registers_per_thread > 128`.

**Why:** compiler couldn't fit all live variables in registers, spilled some to local memory (which is DRAM-backed).

**First-line fix:** `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` on the kernel. This tells the compiler to stay within a register budget.

**Deeper fixes:**
- Reduce the number of live values: recompute values instead of caching, split the kernel into two.
- Move per-thread arrays to shared memory with explicit indexing.

**Exceptions:**
- Large fused kernels (FlashAttention) accept some spill in exchange for larger savings upstream.

**Cross-ref:** Blackwell principle 6.

---

## Pattern L — FP64 used unintentionally

**Signals:**
- `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active > 0` in a kernel that "should" be FP32.

**Why:** C/C++ floating-point literals (`1.0`, `0.5`, `3.14`) default to `double`. A `float x = a + 1.0 * b;` promotes `a + 1.0*b` to double.

**First-line fix:** add `f` suffix to all literals: `1.0f`, `0.5f`, `3.14f`. Add `__expf` / `__logf` / `__sinf` variants for transcendentals.

**Cross-ref:** Blackwell principle 8.

---

## Pattern M — Pipeline bubbles (no compute/memory overlap)

**Signals:**
- PM timeline of `sm__throughput` and `dram__throughput` shows a sawtooth (high compute ↔ high DRAM alternating).
- `long_scoreboard` stalls high but DRAM throughput also high.

**Why:** kernel loads a tile, computes on it, loads next tile — single-buffered.

**First-line fix:** double-buffer. Use two shared-memory tiles; while computing on tile A, load tile B. `__syncthreads` between phases.

**Deeper fixes:**
- Multi-stage pipeline (3-4 stages on Blackwell — see Blackwell principle 15). Use `cp.async` / TMA for async loads.

**Cross-ref:** Blackwell principle 15.

---

## Pattern N — Warp divergence

**Signals:**
- `smsp__thread_inst_executed_per_inst_executed.ratio < 32` (far from the 32 ideal).
- Branch efficiency metric low in `--page details`.
- Divergent branches cluster on specific source lines.

**Why:** lanes in a warp take different paths at a branch; hardware serializes.

**First-line fix:**
- Rearrange so all lanes in a warp take the same branch. Sort / partition data if possible.
- Convert `if (cond) a else b` to branchless `mask * a + (1-mask) * b` — cheap if both sides are cheap.

**Exceptions:**
- Tree reductions in warps (last few steps have half / quarter / ... active). Use `__shfl_down_sync` to handle cleanly.
- Boundary handling (a few warps at tensor edge) — not worth fighting.

**Cross-ref:** Blackwell principle 5.

---

## Pattern O — Serial dependency chain (few warps, fixed-latency stalls)

**Signals:**
- `wait` and `short_scoreboard` lead the per-PC samples, while `long_scoreboard` is small: warps wait on their own fixed-latency and shared-memory / MUFU results, not on global memory.
- `smsp__issue_active.avg.pct_of_peak_sustained_active` low (e.g. < 40%) with only 1-2 warps per scheduler doing the math (`smsp__warps_active.avg.per_cycle_active`; roles of warp-specialized kernels count separately).
- The hot lines are in a loop whose iterations depend on each other (a recurrence, a scan, a state update), and DRAM throughput is low.
- Runtime scales with the length of the chain (sequence length, number of steps), not with the amount of parallel work: halving the heads or batch barely changes it.

**Why:** each step needs the previous step's result, and with one math warp per scheduler there is nothing to issue while it waits, so latency, not throughput, sets the time. Often combined with Pattern A (one CTA per independent chain, fewer chains than SMs).

**First-line fix:** give each scheduler independent work:
- **Split the chain's state across more warps or CTAs** (e.g. along a dimension each step updates independently) so several chains interleave.
- **Keep the state in registers (or TMEM)** rather than reloading it from shared memory every step: a shared-memory round trip per step is on the critical path.

**Deeper fixes:**
- Fewer, larger steps (a bigger chunk per iteration), trading parallel work inside a step for fewer serial steps.
- Move work that does not depend on the chain (precomputation, the inputs of the next step) off the critical path into other warps.

**Exceptions:**
- Chains that are short or already overlapped by many independent CTAs: then other patterns dominate.

**Cross-ref:** Pattern E (the global-memory variant), Pattern A, Pattern J.

---

## Ranking template for the final report

When you hand back an optimization plan, rank by `(expected speedup) × (effort ratio)`. NCU's `Est. Speedup` is a starting estimate: check it against the counters it rests on (it can be far off for small grids, bank conflicts or L2 compression) before you rank by it.

```
Priority 1: <pattern> — <concrete fix>
  Evidence: <metric value(s)>
  NCU Est. Speedup: X%
  Effort: <low / medium / high>
  Why now: <reason this is the highest-leverage fix>

Priority 2: ...
```

A good rule of thumb: at most 3-5 priorities in the plan. More than that dilutes the signal, and priorities > 5 usually contribute < 5% speedup each.
