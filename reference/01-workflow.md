# Profiling Workflow — End-to-End

This is the complete checklist from "user asks to profile" to "final report". Every step has a short rationale and a pointer to the detailed doc.

---

## Phase 0 — Create a new run directory

**Always start here.** See [`00-directory-layout.md`](00-directory-layout.md) for the full convention.

```bash
# <profile_root>: the directory your project keeps profiling artifacts in (else one you create, named in the report)
export PROFILE_RUN_DIR=<profile_root>/<descriptive_run_name>        # e.g. <kernel>_v1_baseline
mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}
```

- Pick a new, descriptive name for this run. Never reuse an existing directory.
- If you're profiling a new version of a kernel you've profiled before, that's a **new** run (e.g. `<kernel>_v2_optimized/`, not overwriting `<kernel>_v1_baseline/`).
- If you're profiling the same version against a different workload, that's also a new run — or, at minimum, each workload's `.ncu-rep` gets a distinct tag and the analysis scripts are kept separate.

Every artifact produced in subsequent phases is written **only** under `$PROFILE_RUN_DIR`. Never into a sibling run's directory.

---

## Phase 0.5 — Frame the problem (before any tools)

Before typing any commands, answer these in your head (or in a few lines of `REPORT.md`):

1. **What kernel(s) am I profiling?** Get the exact kernel name or regex. Kernels are often templated (`foo_kernel<8, 256>`); ncu's `-k "regex:..."` matches the function name (`foo_kernel`) by default, and the full demangled name with `--kernel-name-base demangled`.
2. **Which workload / input shape?** If the kernel takes variable-sized inputs, pick a **specific** real workload — don't invent shapes. If the user has multiple representative shapes, profile the hottest one first; profile others only if the first reveals nothing.
3. **Which dispatch path?** Many production kernels branch on input shape or other runtime values to pick different grid configs or template instantiations. Profile each *active* dispatch path separately — treating them as one kernel will average out the real patterns.
4. **What question am I answering?** "Why is this slow?" is too vague. Better: "At shape X, is the kernel latency-bound or bandwidth-bound?" or "We spent 2 weeks on optimization Y — did it actually help?"
5. **What is the baseline?** If there's a reference implementation (torch, cuBLAS, a previous version), profile it too for comparison.

If any of 1-4 are unclear, settle them before profiling: ask whoever set the task if you can, otherwise choose and record the choice and the reason in `REPORT.md`. Profiling the wrong thing wastes an hour.

---

## Phase 1 — Environment check

Steps 2 and 5, and every later `ncu` collection, need the GPU; use the environment's approved GPU execution path (see [Where GPU work runs](03-collection.md#where-gpu-work-runs)). The rest runs anywhere.

```bash
# 1. ncu CLI is available
ncu --version   # this skill is written against Nsight Compute 2026.3 (CUDA 13.2 bundles 2026.1.1)

# 2. GPU is visible, and which one (the compute capability picks the -arch, 02-harness-guide.md)
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv

# 3. CUDA compiler is available
nvcc --version  # CUDA Toolkit, used for -lineinfo builds

# 4. ncu_report Python module (parses reports): the helpers find it from the ncu on PATH
python3 <skill>/helpers/ncu_utils.py     # prints where ncu_report was found

# 5. Permissions. ncu usually works without sudo (RestrictProfilingToAdminUsers is 0
#    by default). If you see ERR_NVGPUCTRPERM, see 09-common-issues.md.
ncu --section SpeedOfLight -c 1 <any command that launches a kernel>
```

`<skill>` is this skill's directory. If NCU cannot deploy section files, configure a supported writable cache/location for that installation; do not repurpose the shell's home directory.

---

## Phase 2 — Build a profile target

See [`02-harness-guide.md`](02-harness-guide.md).

**Option A: profile the real code.** When the kernel already has line info — a build with `-lineinfo`, or a prebuilt library that ships it (check with `nvdisasm -g`, 02-harness-guide.md) — profile it in place: the existing binary, or a few-line driver script that calls the library once on a representative input.

**Option B: standalone harness.** Build a small C++ driver that launches your kernel directly. This is the right choice when:

- The kernel lives inside a JIT/template build system (TVM-FFI, PyTorch inline, Triton, CUTLASS JIT) that drops line info and where you can't add `-lineinfo`.
- You want fast iteration on the kernel source — the harness compiles in seconds, vs minutes for rebuilding the whole framework.
- You want precise control over inputs (e.g., load specific workload tensors from the dataset).

Either way, **the kernel must carry line info**. Without it, source-level analysis won't work.

---

## Phase 3 — Collect profiles

After environment preflight and target-guide selection, use one `ncu` invocation per (kernel, dispatch path, representative workload) — **outputs go under `$PROFILE_RUN_DIR/reports/`**. Details in [`03-collection.md`](03-collection.md).

```bash
ncu --set full --import-source yes \
    -k "regex:YOUR_KERNEL_NAME" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" \
    "$PROFILE_RUN_DIR/harness/your_harness" [args]
```

`--set full` membership and measurement availability depend on the installed collector and target. Check source/PM coverage in the resulting report; unavailable measurements remain explicit gaps. Name the output with its `.ncu-rep` extension: without one, ncu 2026.3 writes `.ncu-repz`. A `--set full` run replays the kernel about 50 times: seconds for a short kernel, minutes for a long one. Plan your time budget.

---

## Phase 4 — Extract structured data

Do not eyeball the CLI output. Parse reports in Python so you can compare, aggregate, and archive. See [`04-python-api.md`](04-python-api.md) and use the helpers in [`../helpers/`](../helpers/).

Minimum analysis artifacts to produce:

| Artifact | Tool | What it tells you |
|---|---|---|
| `metrics_evidence_<tag>.json` | `analyze_reports.py` | Target, separate collector/reader provenance, raw units and value states |
| `metrics_key_<tag>.txt` | `analyze_reports.py` | ~100 curated metrics (launch geom, clock, SOL, occupancy, stalls, sectors) |
| `metrics_all_<tag>.json` | `analyze_reports.py` | All metrics actually present, archive for later |
| `rules_<tag>.txt` | `analyze_reports.py` | NCU's rule-engine results, highest `Est. Speedup` first |
| `compare_<a>_vs_<b>.txt` | `analyze_reports.py` | Side-by-side metric comparison between workloads / versions |
| `stall_hotspots_<tag>.txt` | `extract_stall_hotspots.py` | Top source lines ranked by stall samples (`--binary`: by the kernel's own lines) |
| `pm_timeline_<tag>.txt` | `plot_timeline.py` | ASCII time-series plots — reveals tail effect visually |
| `details_<tag>.txt` | `ncu --import ... --page details` | NCU's details page, as the GUI shows it |

Save everything under `$PROFILE_RUN_DIR/analysis/`. The user will want to re-inspect these; if two runs mix artifacts, you've already failed.

---

## Phase 5 — Diagnose

Work through the six analysis dimensions — see [`05-analysis-dimensions.md`](05-analysis-dimensions.md):

1. **SM occupancy & wave structure** — are enough blocks launched to fill the chip? Is occupancy register- / shared-mem- / block-limited?
2. **Thread-block balance (tail effect)** — do per-SM active cycles match? Does the PM timeline show a clean drop or a gradual tail?
3. **Instruction-level stall analysis** — what stall reason dominates? Which source line generates it?
4. **Tensor Core utilization** — if this is a GEMM-ish kernel, are tensor cores actually being used?
5. **SM utilization timeline** — flat high, flat low, periodic waves, gradual tail?
6. **Memory access pattern** — sectors/request, L1/L2 hit rates, DRAM throughput, register spill.

For each dimension, write down the observed signal *and the specific metric value* that produced it. "Kernel is memory bound" is useless; something like "`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed = X%` (well below peak) shows the kernel is *not* DRAM-bandwidth-bound — the `long_scoreboard` stall rate of Y% says it's latency-bound on L1" is diagnosis. Fill in X and Y from your own report.

Then consult [`06-diagnosis-playbook.md`](06-diagnosis-playbook.md) which maps observed patterns to likely causes and concrete fixes.

---

## Phase 6 — Write the report

Structure described in [`07-report-template.md`](07-report-template.md). Key elements:

1. **Setup section**: exactly how you profiled (harness path, workloads, ncu commands, metric-name caveats). Required for reproducibility.
2. **Headline numbers**: duration, SM throughput, DRAM throughput, occupancy, tensor core usage. A table on the first page.
3. **Per-dimension analysis** with evidence (metric values + NCU rule text).
4. **Optimization directions** ranked by expected impact (use NCU's `Est. Speedup: X%` when available, after checking each against the counters it rests on — some are far off).
5. **Confidence & caveats**.

Keep the report short enough that a busy reader can see the top 3 findings in 30 seconds. Put deep detail in the artifacts, not the prose.

---

## Anti-patterns to avoid

- ❌ **"I ran ncu and it says memory throughput is 14%"** — without naming the metric, workload, and kernel, this is un-actionable. Always give metric + value + what it means.
- ❌ **Profiling with synthetic shapes that don't match real workloads.** A uniform-element batch is a very different problem than a batch with highly skewed per-element work (the latter exposes tail effects the former hides). If the production workload has imbalance, you must profile on an imbalanced workload.
- ❌ **Dumping the full NCU CLI output into the report.** It's noisy, narrow-formatted, and has no interpretation. Extract the numbers, cite the source, add your reading.
- ❌ **Proposing optimizations without evidence.** "Maybe we should use shared memory" is not a profiling result. A real proposal cites a specific source line, its stall-sample count, the relevant NCU rule's `Est. Speedup`, and the mechanism of the fix — e.g. "line L's global-load instruction accounts for N% of `long_scoreboard` samples; NCU reports the access pattern is non-coalesced with M% excess sectors; reshaping the per-thread index from stride-K to contiguous should eliminate most of those stalls."
- ❌ **Missing the #1 finding because you got distracted by a smaller one.** Rank findings by impact. Tail effects and SM idle time often dwarf coalescing issues; fix the big one first.
