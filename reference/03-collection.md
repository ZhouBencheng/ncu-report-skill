# Profile Collection Commands

This document lists the exact `ncu` commands you should run, in what order, and what each flag does. They are written against Nsight Compute 2026.1; `ncu --help`, `ncu --list-sets` and `ncu --list-sections` show what your version has.

---

## Prerequisites recap

- Line info in the kernel: `-lineinfo` in the compile flags, or a prebuilt binary that has it (see `02-harness-guide.md`).
- `ncu` available on PATH.
- Writable `$HOME` so ncu can cache section files.
- Kernel name known (`cuobjdump -res-usage <binary> | grep Function`, or the survey below).

Every `ncu` command below that runs a program needs the GPU; `ncu --import` does not.

Quick permission test:
```bash
ncu --section SpeedOfLight -k "regex:YOUR_KERNEL_NAME" -c 1 ./harness [args]
# If you see ERR_NVGPUCTRPERM: see 09-common-issues.md.
# If you see the SpeedOfLight table and "regex" matched, you're good.
```

---

## Recipe 0: Survey — which kernels run, and how long each takes

Before profiling in depth, list every kernel a call launches with its duration and clock (cheap: one pass per kernel):

```bash
ncu --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg.per_second \
    -o "$PROFILE_RUN_DIR/reports/survey_<tag>" ./harness [args]
```

Each launch is one entry with its demangled name: pick the kernel and the regex for `-k` from here. `analyze_reports.py` on the survey report writes one set of files per kernel.

---

## Recipe 1: Full profile (the one report the analysis needs)

Collects every standard section — SOL, occupancy, memory, compute, scheduler, warp state, the per-PC stall samples (SourceCounters) and the PM-sampling timelines — plus the stall-reason timelines:

```bash
ncu --set full \
    --section PmSampling_WarpStates \
    --import-source yes \
    -k "regex:KERNEL_REGEX" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>" \
    ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--set full` | Every built-in section, including `SourceCounters` and `PmSampling` (ncu 2026.1) |
| `--section PmSampling_WarpStates` | Time-series of warp stall states (not in `full`) |
| `--import-source yes` | Embeds the CUDA source files into the report, so the source view works where the source is not (another machine, a container). The files are read where ncu runs, at collection only (it cannot be added to an existing report): when the paths recorded by the build do not exist there, pass `--source-folders DIR,...` with directories visible to that machine or container |
| `-k "regex:..."` | Only profile kernels whose demangled name matches. Reduces replay count. |
| `-c 1` | Only profile the first matching kernel launch. Avoids duplicates when the kernel is called in a loop. |
| `-o "$PROFILE_RUN_DIR/reports/full_<tag>"` | Output path — `.ncu-rep` is appended automatically. |

The kernel is replayed dozens of times (one pass per metric group): seconds for a short kernel, minutes for a long one.

---

## Recipe 2: Source counters only (fast re-check of hot lines)

Per-PC stall sampling alone, a few passes. Use it after a code change, when only the hot lines matter:

```bash
ncu --section SourceCounters \
    -k "regex:KERNEL_REGEX" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/source_<tag>" \
    ./harness [args]
```

`extract_stall_hotspots.py` reads either this report or the full one. There is no `source` set in ncu 2026.1: `--set source` is not rejected, it silently collects the default `basic` set.

---

## Recipe 3: Details page (quick rule summary)

No need to collect again — just import an existing `full_<tag>.ncu-rep`:

```bash
ncu --import "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page details > "$PROFILE_RUN_DIR/analysis/details_<tag>.txt"
```

This produces NCU's human-readable details page, including the built-in rule-engine suggestions. Each rule looks like:

```
OPT   Est. Speedup: <pct>%
      <description of the pattern NCU detected, e.g. "On average, each warp
      spends N cycles stalled waiting for a scoreboard dependency on a L1TEX
      operation. Find the instruction producing the data being waited upon
      to identify the culprit.">
```

**Read the rules first** — they point at where to look. `analyze_reports.py` writes the same results sorted by estimated speedup (`rules_<tag>.txt`). Check each estimate against the counters before ranking by it (see `09-common-issues.md`).

---

## Recipe 4: CSV / raw export (scripting)

```bash
# Full metric table as CSV — one row per kernel launch, one column per metric
ncu --import "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page raw --csv > "$PROFILE_RUN_DIR/analysis/raw_<tag>.csv"

# Source page as text (CUDA source with per-line counters; --print-source sass for SASS)
ncu --import "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page source --print-source cuda > "$PROFILE_RUN_DIR/analysis/source_<tag>.txt"
```

Most of the time you don't need these — the Python API (`04-python-api.md`) is easier. But CSV is handy for quick `grep`/`awk` one-liners.

---

## Recipe 5: Targeted metrics only (fast)

If you already know which metrics you want (e.g., you're re-running after a code change and only want to check if the fix worked), collect just those:

```bash
ncu --metrics \
    gpu__time_duration.sum,\
    sm__cycles_elapsed.avg.per_second,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    dram__bytes_read.sum.pct_of_peak_sustained_elapsed,\
    l1tex__t_sector_hit_rate.pct,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  -k "regex:KERNEL_REGEX" -c 1 \
  ./harness [args]
```

Takes one or two replay passes. Prints a table directly to stdout. No `-o` output file. `--metrics` also reaches metrics no section collects (`dram__bytes.sum`, for one; see `08-metric-names.md`).

---

## Recipe 6: A/B comparison (before vs after optimization)

```bash
# Before
ncu --set full -k "regex:my_kernel" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/v1" ./harness_v1 [args]

# After
ncu --set full -k "regex:my_kernel" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/v2" ./harness_v2 [args]
```

Then use `analyze_reports.py` with multiple tags to produce a side-by-side comparison, or manually:

```python
from ncu_utils import load_action           # <skill>/helpers on sys.path
a1 = load_action(f"{run_dir}/reports/v1.ncu-rep")
a2 = load_action(f"{run_dir}/reports/v2.ncu-rep")
t1 = a1["gpu__time_duration.sum"].value()
t2 = a2["gpu__time_duration.sum"].value()
print(f"Speedup: {t1/t2:.2f}x")
```

Compare reports collected at the same clock (`--clock-control`, below).

---

## What each `--set` contains

```bash
ncu --list-sets           # list all sets
ncu --list-sections       # list all sections
```

ncu 2026.1:

| Set | Sections included | Use when |
|---|---|---|
| `basic` (default) | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution | Smoke test — is ncu working at all? |
| `detailed` | basic + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis (+ chart), SourceCounters, SpeedOfLight_RooflineChart, Tile | Middle ground |
| `full` | everything above + InstructionStats, MemoryWorkloadAnalysis_Tables, NumaAffinity, Nvlink_Tables / _Topology, PmSampling, SchedulerStats, the hierarchical roofline charts, WarpStateStats | First-pass profile. Always start here. |
| `pmsampling` | PmSampling, PmSampling_WarpStates | Timelines only |
| `roofline` | SpeedOfLight + the roofline charts, WorkloadDistribution | Roofline only |
| `nvlink` | Nvlink, Nvlink_Tables, Nvlink_Topology | Multi-GPU traffic |

---

## Common section additions

```bash
# Stall-reason timelines (PmSampling itself is in --set full)
--section PmSampling_WarpStates

# NVLink bandwidth (if profiling multi-GPU)
--section Nvlink
```

---

## Profiling multiple kernel launches

If the kernel is called multiple times and you want different iterations:

```bash
# Skip the first 5 invocations, then profile 3 consecutive launches
ncu -k "regex:my_kernel" -s 5 -c 3 -o report ./harness
```

`-s N` skips the first N matches. `-c N` limits to N matches. Useful for ignoring warmup or focusing on a steady-state iteration. For a program that warms up first (JIT, autotuning), `--profile-from-start off` plus a profiler range around the call you want (`cudaProfilerStart/Stop`, `torch.cuda.profiler.start()/stop()`) profiles only that call.

A report with several launches holds one action per launch; the helpers analyze each (tags `<tag>.<i>_<function>`), and `--kernel REGEX` picks some.

---

## GPU clocks

ncu fixes the clock itself: it locks the GPU clock for every profiled kernel and restores it after, so durations and
throughputs are comparable between runs. The default is `--clock-control boost` in ncu 2026.1 and `base` in older
versions (`ncu --help` shows it); on a B300, `base` (~1.07 GHz) made kernels ~1.7x slower than `boost` (~1.9 GHz).
Name the setting in the command so the report says what it ran at. No outer lock is needed.

To profile at a fixed clock of your own (to match the clock a timing A/B ran at), lock it around the command and pass
`--clock-control none`; without it, ncu replaces your lock with its own:

```bash
# check with nvidia-smi -q -d CLOCK; an environment that locks clocks for you replaces these two lines
sudo nvidia-smi -lgc <mhz>,<mhz>
ncu --clock-control none --set full ... ./harness [args]
sudo nvidia-smi -rgc
```

Read the clock a report ran at from `sm__cycles_elapsed.avg.per_second` (in `metrics_key_<tag>.txt`), and give it in the report.

---

## Gotchas

- **Profile run time blows up with `--set full`**: that's normal, each replay is a full kernel execution.
- **`regex:...` doesn't match anything**: the regex matches the demangled name. Templates produce names like `void my_kernel<(int)8, (int)256>(...)`; list them with the survey (Recipe 0) or `cuobjdump -res-usage ./harness | grep Function | cu++filt`.
- **Report file is empty / 0 KB**: profile terminated before the kernel launched. Usually means `-k` regex didn't match, or the harness crashed.
- **PM Sampling returns nothing**: check the report came from `--set full` (or `--section PmSampling`) and the GPU isn't a vGPU (vGPU doesn't support PM sampling). `plot_timeline.py --list` shows the series a report has.
- **"Could not deploy stock section files to $HOME"**: set `HOME` to a writable directory first.
