---
name: ncu-report-skill
description: Profile CUDA kernels with Nsight Compute on NVIDIA Thor (sm_110), B200 (sm_100), and Jetson AGX Orin (sm_87, CC 8.7). Use when the user asks to profile a kernel, analyze its performance, diagnose bottlenecks, read an ncu report, or write an optimization plan — including variants in Chinese ("profile 一下", "为什么慢", "ncu 报告").
---

# Skill: CUDA Kernel Profiling (Thor / B200 / Orin / Nsight Compute)

**When to use:** user asks to profile a CUDA kernel, analyze its performance, find its bottlenecks, or write an optimization plan based on Nsight Compute data. Triggers include: "profile X", "为什么这个 kernel 慢", "ncu report 说...", "下一步怎么优化", "帮我看一下这份 ncu 报告".

**Target hardware (this repo):** NVIDIA **Thor (sm_110, CC 11.0)**, **B200 (sm_100, CC 10.0)**, and **Jetson AGX Orin (sm_87, CC 8.7)**. The resource facts *148 SMs, 192 GB HBM3e* are **B200-specific** — do not assume them for Thor or Orin. Most advice below is generic; B200-specific notes are explicitly marked. On any target, record the actual device resource facts and distinguish runtime compute capability from the compiled target. For other GPUs, verify available metrics and hardware-specific assumptions.

**Orin (sm_87) entry point:** on Jetson AGX Orin the collector and the report reader are **two separate tools with separate versions** (collector NCU 2025.2.1 on-device; reader aarch64 `ncu-report==2026.3.1`). Start from [`references/10-orin-collector-reader.md`](references/10-orin-collector-reader.md) for the tested pairing, the runnable collect/read recipes, and the `--profile orin` curated view; the analyzer auto-selects the Orin profile only on a qualified `Orin`/CC 8.7 identity (else `generic`, never B200). The device identity, supported/unsupported instruction scope, and bounded execution evidence live in the companion KernelWiki pages `KernelWiki/wiki/hardware/orin.md` (`hw-orin`) and `KernelWiki/wiki/hardware/ampere-mma-async-copy.md` (`hw-ampere-mma-async-copy`).

**Thor (SM110) preflight — before building/collecting:** record the selected `nvcc` / `ptxas` / `ncu` paths and versions, the *actual* device resource facts (SM count, resident warps/threads, shared-memory and L2 limits, SKU memory — never copied from B200), and keep runtime **CC 11.0** distinct from the compiled native/virtual target. Reuse the version/set/section preflight in [`references/03-collection.md`](references/03-collection.md) and the companion KernelWiki Thor page (`KernelWiki/wiki/hardware/thor.md`) and its evidence registry (`KernelWiki/data/thor-evidence.yaml`); select `sm_110f` or `sm_110a` only from instruction/compiler evidence, never from runtime CC alone. Retained B200 guidance stays labeled B200.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing CUDA kernels are under-performing for exactly one reason that ncu can tell you in 10 seconds. Don't invent hypotheses before you have the report. Don't start coding a fix before you've matched the observed pattern to a known diagnosis. Don't write a wall of suggestions — rank them by evidence and expected impact.

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** under `profile/<run_name>/` at the repo root — **one directory per run**, never reuse an existing one. Each run contains its own `harness/`, `reports/`, `analysis/`, and `REPORT.md`. This rule is mandatory in this repo. See [`references/00-directory-layout.md`](references/00-directory-layout.md).

1. **Decide what you're profiling.** What inputs? Which dispatch path? What question do you want answered? If the kernel takes variable-sized inputs (variable seq lengths, variable batch sizes), you must pick specific representative shapes from the user's workload — don't profile with arbitrary inputs.

2. **Build a standalone harness** unless the user is profiling through their existing binary. Harnesses compile in seconds, run the kernel in isolation, and let you use `-lineinfo` cleanly so ncu can map SASS back to source. Compile into `profile/<run_name>/harness/`. See [`references/02-harness-guide.md`](references/02-harness-guide.md) and the template in [`assets/harness_template.cu`](assets/harness_template.cu).

3. **Check available sets and sections, then collect profiles**: `--set full` (with available `PmSampling` sections) for the overview, and `--section SourceCounters` for per-line stall attribution. Add `--set source` only if `ncu --list-sets` lists it. Source and PM availability must be confirmed in the report; unavailable measurements are not zero. Write outputs to `profile/<run_name>/reports/`. See [`references/03-collection.md`](references/03-collection.md). **On Jetson AGX Orin (sm_87)** the collector/reader split and exact runnable recipes are in [`references/10-orin-collector-reader.md`](references/10-orin-collector-reader.md).

4. **Parse with `ncu_report`** Python module — not by eye-balling the CLI. Write analysis outputs to `profile/<run_name>/analysis/`. Use the scripts in [`scripts/`](scripts/). See [`references/04-python-api.md`](references/04-python-api.md). For Orin reports, read with the aarch64 `ncu-report==2026.3.1` reader and the `--profile orin` curated view per [`references/10-orin-collector-reader.md`](references/10-orin-collector-reader.md).

5. **Work through the six analysis dimensions.** See [`references/05-analysis-dimensions.md`](references/05-analysis-dimensions.md). Every one matters, but on any given kernel only 1–2 will dominate.

6. **Match patterns to the diagnosis playbook.** See [`references/06-diagnosis-playbook.md`](references/06-diagnosis-playbook.md). It maps NCU signal → likely cause → concrete fix, with example counts for "how big is this".

7. **Write the report** at `profile/<run_name>/REPORT.md` with evidence-backed recommendations, ranked by expected impact. See [`references/07-report-template.md`](references/07-report-template.md).

---

## File index

### Reference docs (read these when you need details)

| File | Purpose |
|---|---|
| [`references/00-directory-layout.md`](references/00-directory-layout.md) | **Read first.** Directory / naming conventions — one run = one subdirectory, no cross-contamination |
| [`references/01-workflow.md`](references/01-workflow.md) | End-to-end checklist from "user request" to "final report" |
| [`references/02-harness-guide.md`](references/02-harness-guide.md) | When and how to build a standalone harness (mandatory for TVM-FFI, PyTorch kernels, JIT-compiled code) |
| [`references/03-collection.md`](references/03-collection.md) | ncu command recipes: full, source-level, PM sampling, custom sections |
| [`references/04-python-api.md`](references/04-python-api.md) | `ncu_report` Python API patterns with copy-pasteable code |
| [`references/05-analysis-dimensions.md`](references/05-analysis-dimensions.md) | Six analysis dimensions: occupancy, balance, stalls, tensor core, timeline, memory |
| [`references/06-diagnosis-playbook.md`](references/06-diagnosis-playbook.md) | Pattern → diagnosis → fix. Merges Blackwell programming principles with NCU signals |
| [`references/07-report-template.md`](references/07-report-template.md) | How to structure the final report |
| [`references/08-b200-metric-names.md`](references/08-b200-metric-names.md) | sm_100 metric names vs older GPUs — many common names are different |
| [`references/09-common-issues.md`](references/09-common-issues.md) | Permissions, PM sampling gaps, TVM-FFI / PyTorch gotchas |
| [`references/tool-usage.md`](references/tool-usage.md) | How to run scripts and copy harness assets |
| [`references/10-orin-collector-reader.md`](references/10-orin-collector-reader.md) | **Jetson AGX Orin (sm_87).** Collector/reader version split (collector NCU 2025.2.1; reader aarch64 `ncu-report==2026.3.1`), runnable collect/read recipes, and the `--profile orin` curated view |
| [`references/blackwell-cuda-programming.md`](references/blackwell-cuda-programming.md) | Blackwell-specific programming principles and checklists |

### Scripts (executable tools and shared module)

| File | Purpose |
|---|---|
| [`scripts/analyze_reports.py`](scripts/analyze_reports.py) | Extract key metrics, produce side-by-side comparisons |
| [`scripts/extract_stall_hotspots.py`](scripts/extract_stall_hotspots.py) | Per-line stall aggregation via `action.source_info(pc)` |
| [`scripts/plot_timeline.py`](scripts/plot_timeline.py) | ASCII PM-sampling timeline plotter — makes tail effect visible |
| [`scripts/list_flashinfer_workloads.py`](scripts/list_flashinfer_workloads.py) | Browse a flashinfer-trace dataset — shape histograms, filter by axis, resolve safetensors paths for specific UUIDs |
| [`scripts/ncu_utils.py`](scripts/ncu_utils.py) | Shared Python helpers: safe metric access, per-instance extraction, report loading |

### Assets (code resources for profiling harnesses)

| File | Purpose |
|---|---|
| [`assets/harness_template.cu`](assets/harness_template.cu) | Standalone harness template — paste your kernel, fill in input allocation, done |
| [`assets/safetensors_loader.h`](assets/safetensors_loader.h) | Header-only safetensors reader (no external deps) for loading real workload tensors |

---

## Critical lessons (don't skip)

1. **The stock `ncu_profile_skill.md` metric names don't all work on B200.** Names like `smsp__inst_executed_op_global_ld.sum`, `dram__bytes.sum`, `l1tex__average_t_sectors_per_request*.ratio` return `None` on sm_100. Use the sm_100 names in [`references/08-b200-metric-names.md`](references/08-b200-metric-names.md) or enumerate via `action.metric_names()`.

2. **Always compile with `-lineinfo`.** Without it, ncu's source view is blank and you cannot do per-line stall analysis. If you can't add `-lineinfo` to the build system (TVM-FFI, PyTorch inline, JIT), **build a standalone harness** — that's the whole point.

3. **PM sampling is the only way to see tail effects.** Static metrics average over the whole kernel; only the time-series (either `pmsampling:` metrics or the ASCII plotter in `scripts/`) shows the shape of utilization over time.

4. **Load-imbalance on variable-length inputs is often the #1 bottleneck.** If the user's workload has sequences of varying length, per-SM active-cycle variance will often dwarf every other effect. Always check the input distribution.

5. **NCU's rule engine (`--page details`) already does half the work.** Each rule comes with `Est. Speedup: X%`. Read them first — they often point straight at the answer.

6. **Don't delegate understanding.** Run the profiles yourself, open the reports, cite specific metric values. Never write "the profile shows it's memory-bound" — instead, name the two or three metric values that back your conclusion (e.g., "`dram__bytes_read.sum.pct_of_peak_sustained_elapsed` well under 10%, and `long_scoreboard` stalls dominate the pcsamp histogram, so the kernel is **latency-bound on L1**, not DRAM-bandwidth-bound"). Fill in the actual numbers from your report. Specificity is the deliverable.

---

## Companion reference

- [`references/blackwell-cuda-programming.md`](references/blackwell-cuda-programming.md) — Blackwell-specific programming principles and checklists, preserved as a companion reference. Use it when proposing *new* kernel designs; use this skill when diagnosing *existing* kernels.
