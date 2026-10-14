# Profile Directory Layout & Naming

**Read this first, before any collection.** Bad directory layout is the single most common cause of mixing results from different runs, overwriting prior profiles, or losing track of which `.ncu-rep` belongs to which kernel version. The rules below are non-negotiable.

---

## Top-level rule

**All profiling artifacts live under one profile root.** Use the directory your project names for profiling artifacts (look in its README, AGENTS.md or contributing docs); if it names none, create one at the repo root and name it in the report. Never scatter `.ncu-rep` files across random locations. Never put profile artifacts under `solution/`, `src/`, `scripts/`, or other source directories.

```
<repo_root>/
├── <profile_root>/                 ← everything profiling-related lives here
│   ├── <run_1>/
│   ├── <run_2>/
│   └── ...
├── solution/                       ← untouched by profiling
├── src/
└── ...
```

---

## One run = one subdirectory

Every time you profile a kernel — whether it's a new kernel, a new version of the same kernel, or the same kernel on a different workload — **create a new subdirectory under the profile root**. Never write into an existing run's directory.

Rationale:

- Profiles of different implementations of the same kernel must not overwrite each other. If you profile `<kernel>_v1` today and `<kernel>_v2` tomorrow, both reports need to coexist for A/B comparison.
- The harness itself is part of the profile: it encodes which kernel code was compiled, with which flags, against which workload. Keeping the harness source in the run dir pins the provenance.
- Analysis artifacts (`metrics_*.json`, `compare_*.txt`, ASCII plots) are tied to a specific set of `.ncu-rep` files; they must not be mixed.

---

## Run directory naming

Use descriptive, short, snake_case names. Include **what** was profiled and **when/why**, not how.

Good:
```
<kernel>_v1_baseline/
<kernel>_v2_optimized/
<kernel>_v2_optimized_vs_v1/      # for comparison run
moe_fp8_v4_tma_prefetch/
flash_attn_b200_h128_baseline/
```

Bad:
```
test/                   # too vague
run1/                   # meaningless
20260413/               # dates with no context
final/                  # there's never a "final"
```

If you genuinely have multiple runs on the same day for the same kernel/version combo, append a short distinguisher or a date suffix: `<kernel>_v1_baseline_20260413_am` / `<kernel>_v1_baseline_20260413_pm`.

---

## Standard run layout

Inside each run subdirectory, use this structure:

```
<run_name>/
├── REPORT.md                       ← human-readable final report (Markdown)
├── harness/
│   ├── <kernel>_harness.cu         ← the exact source that was compiled (or driver.py for a library)
│   ├── <kernel>_harness            ← compiled binary (with -lineinfo)
│   └── build_command.sh            ← optional: shell script that compiled it
├── reports/
│   ├── full_<tag1>.ncu-rep         ← ncu --set full output (includes source counters + PM sampling)
│   └── full_<tag2>.ncu-rep
└── analysis/
    ├── metrics_all_<tag>.json      ← 2000+ metrics, full archive
    ├── metrics_key_<tag>.{txt,json}← curated key metrics
    ├── rules_<tag>.txt             ← NCU rule results, highest estimated speedup first
    ├── compare_<a>_vs_<b>.txt      ← side-by-side
    ├── details_<tag>.txt           ← ncu --page details dump
    ├── stall_hotspots_<tag>.txt    ← per-line stall aggregation
    ├── pm_timeline_<tag>.txt       ← ASCII time-series
    └── raw_<tag>.csv               ← optional: ncu CSV export
```

Notes:

- `<tag>` is the per-workload / per-dispatch-path label, e.g. `path_a_shapeA`, `path_b_shapeB`. Pick tags that are short and name the representative workload, not the file UUID.
- If you profile only one tag, you can omit the tag suffix from filenames. But as soon as you profile a second, backfill the tag to avoid ambiguity.
- A report that holds several kernels (no `-k` / `-c 1`) gives one set of analysis files per kernel, tagged `<tag>.<i>_<function>`.
- Any one-off analysis script you write for this run goes in `analysis/` next to its output, so the run is self-contained and archivable.

---

## Comparing two runs

For A/B comparisons (optimization-before vs after, or two dispatch variants on the same build), create a comparison run that *references* both underlying runs:

```
<kernel>_v2_vs_v1/
├── REPORT.md                       ← describes both runs + the comparison
└── analysis/
    ├── compare.py                  ← loads reports from the two runs below
    ├── compare_key_metrics.txt     ← side-by-side on key metrics
    └── compare_stalls.txt          ← side-by-side on stall breakdown
    (No ncu-rep files — they live in the referenced runs)
```

In `compare.py`, locate both referenced runs relative to the comparison run, so the profile root can move:
```python
ROOT = Path(__file__).resolve().parents[2]      # the profile root
V1_DIR = ROOT / "<kernel>_v1_baseline"
V2_DIR = ROOT / "<kernel>_v2_optimized"
```

The comparison run does not re-profile; it only produces comparison artifacts and prose.

---

## What does NOT go in a run directory

- `.ncu-rep.old` backup files — if you need a prior version, you should have made it a separate run.
- Temporary scratch files — use a temp directory for those.
- The dataset / workload files themselves — these belong in the project's shared dataset directory. Reference them from scripts by a path argument or the dataset's own environment variable, not a copy.
- Compiler intermediates (`*.o`, `*.d`). Put them under `harness/build/` or just rely on rebuilding from source.
- ncu's cache directory (it writes section files under `$HOME`) — keep `HOME` pointing outside the run dir.

To keep run dirs out of git, a `.gitignore` in the profile root:
```
*/
!README.md
```

Or, to track a few canonical runs, ignore only the data-heavy parts:
```
*/reports/
*/analysis/metrics_all_*.json
*/analysis/raw_*.csv
*/harness/*_harness          # binary only, keep the .cu
```

---

## Environment variable convention (optional but recommended)

Scripts and ncu invocations should pick up the run directory from a single variable, so they're easy to redirect to different runs:

```bash
export PROFILE_RUN_DIR=<profile_root>/<kernel>_v1_baseline
mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}

# build harness (sm_100a: B200; sm_103a: B300 — see 02-harness-guide.md)
nvcc -O2 -std=c++17 -lineinfo -arch=sm_100a \
     harness.cu -o "$PROFILE_RUN_DIR/harness/kernel_harness"

# run ncu (where the GPU is)
ncu --set full --import-source yes \
    -k "regex:my_kernel" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" \
    "$PROFILE_RUN_DIR/harness/kernel_harness" [args]

# parse
python3 <skill>/helpers/analyze_reports.py --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag>
```

`<skill>` is this skill's directory. The report helpers in `../helpers/` (`analyze_reports.py`, `extract_stall_hotspots.py`, `plot_timeline.py`) take `--run-dir` and write under `<run-dir>/analysis/`.

---

## Checklist before starting a profile run

1. `mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}` — a new run name, the three subdirs up front.
2. Copy or write the harness or driver into `harness/`.
3. Compile into the same dir with `-lineinfo` (if you compile at all).
4. Run ncu with `-o "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep"`.
5. Put analysis outputs and one-off scripts under `analysis/`.
6. Write `REPORT.md` at `$PROFILE_RUN_DIR/REPORT.md`.
7. Before starting a *new* run, go back to step 1 with a new name — never write into the existing one.
