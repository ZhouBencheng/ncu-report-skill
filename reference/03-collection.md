# Profile Collection Commands

The operational recipes below are shared: selecting launches, importing source, warm-up, replay and exporting reports. Select sections and clock behavior from the actual device/collector. Collector and report-reader versions are separate facts.

After creating a new run directory and setting `PROFILE_RUN_DIR`, discover the environment:

```bash
python3 helpers/preflight.py --ncu /actual/path/to/ncu --device 0 \
    --output "$PROFILE_RUN_DIR/preflight.json"
```

This read-only helper identifies the device, queries version/help/sets/sections/metrics and proposes `collection_options`; it launches no kernel and changes no clocks. Inspect those options, then read only the selected [target guide](../SKILL.md#select-the-environment-before-loading-architecture-advice). The examples below assume their named sections/options were found. It proposes `--clock-control none` for identified Orin/Thor targets and whenever L4T is observed, even if unavailable `nvidia-smi` leaves the exact device generic. L4T evidence does not invent a device identity. Discovery is not a successful permission test.

For an existing report, skip collection and use `helpers/analyze_reports.py`; it selects the target from that report, with generic fallback for unknown identity.

## Where GPU work runs

Only the collector and the program it launches need the GPU. Use the environment's approved GPU broker, scheduler, container or remote-execution path rather than assuming the current shell owns a GPU. Keep the run directory visible to collection and analysis, and use paths valid in the collector's environment. Harness builds for an explicit target, `ncu --import`, and the Python analysis helpers can run without a GPU. Copy only the sanitized reports or derived artifacts that the authorization permits; do not embed private host identity in committed evidence.

---

## Prerequisites recap

- Line info in the kernel: `-lineinfo` in the compile flags, or a prebuilt binary that has it (see `02-harness-guide.md`).
- `ncu` available on PATH.
- A writable collector cache/configuration location when the installed NCU requires one; do not repurpose the shell's home directory.
- Kernel name known (`cuobjdump -res-usage <binary> | grep Function`, or the survey below).
- ncu launches the program itself, not a launcher around it. `ncu ... uv run python app.py` hangs (no kernel is ever profiled, in ncu 2026.1 and 2026.3 alike): put ncu inside the launcher, `uv run ncu ... python app.py`, or give ncu the interpreter, `ncu ... .venv/bin/python app.py`. The same goes for other launchers that start the interpreter as a child process.

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
    -o "$PROFILE_RUN_DIR/reports/survey_<tag>.ncu-rep" ./harness [args]
```

Read it as one table per kernel:

```bash
ncu --import "$PROFILE_RUN_DIR/reports/survey_<tag>.ncu-rep" --print-summary per-kernel --print-kernel-base function
```

Each entry has the kernel's function name (what `-k` matches), its launch shape, the number of invocations and the metrics: pick the kernel and the regex for `-k` from here.

---

## Recipe 1: Overview profile when the installed full set is suitable

A full set usually provides a broad overview. Its exact sections and PM/timed-warp coverage depend on collector and target; check the collected report rather than assuming every measurement is present:

```bash
ncu --set full \
    --import-source yes \
    -k "regex:KERNEL_REGEX" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" \
    ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--set full` | Installed full set; inspect its membership and warnings |
| `--import-source yes` | Embeds the CUDA source files into the report, so the source view works where the source is not (another machine, a container). The files are read where ncu runs, at collection only (it cannot be added to an existing report): when the paths recorded by the build do not exist there, pass `--source-folders DIR,...` with directories visible to that machine or container |
| `-k "regex:..."` | Only profile kernels whose name matches: the function name by default (`my_kernel`), the full demangled name (`void my_kernel<(int)8>(float *)`) with `--kernel-name-base demangled`. Reduces replay count. |
| `-c 1` | Only profile the first matching kernel launch. Avoids duplicates when the kernel is called in a loop. |
| `-o "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep"` | Output path. Give the extension: without one, ncu 2026.3 appends `.ncu-repz` (2026.1: `.ncu-rep`); both load the same in ncu and the helpers. |

The kernel is replayed dozens of times (one pass per metric group): seconds for a short kernel, minutes for a long one.

---

## Recipe 2: Source counters only (fast re-check of hot lines)

Per-PC stall sampling alone, a few passes. Use it after a code change, when only the hot lines matter:

```bash
ncu --section SourceCounters \
    -k "regex:KERNEL_REGEX" \
    -c 1 \
    -o "$PROFILE_RUN_DIR/reports/source_<tag>.ncu-rep" \
    ./harness [args]
```

`extract_stall_hotspots.py` reads either this report or the full one. Do not assume a `source` set exists: inspect `--list-sets`, or select the available `SourceCounters` section explicitly. If no source measurements are available, state that limitation and continue with available static metrics.

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

# Source page as text: CUDA lines interleaved with their SASS and its per-instruction counters
# (--print-source cuda alone prints the source without counters)
ncu --import "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --page source --print-source cuda,sass > "$PROFILE_RUN_DIR/analysis/source_<tag>.txt"
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
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
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
ncu --set full --import-source yes -k "regex:my_kernel" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/v1.ncu-rep" ./harness_v1 [args]

# After: the same options, so the reports compare
ncu --set full --import-source yes -k "regex:my_kernel" -c 1 \
    -o "$PROFILE_RUN_DIR/reports/v2.ncu-rep" ./harness_v2 [args]
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

## Installed sets and sections

Use `ncu --list-sets` and `ncu --list-sections` for the installed collector. Set membership and PM availability change with version/platform. Detailed NCU 2026.1/2026.3 observations are [version-scoped reference material](architectures/sm100/collection.md), not the contract for every target.

## Common section additions

```bash
# NVLink bandwidth (if profiling multi-GPU; Nvlink_Tables / _Topology are in --set full)
--section Nvlink

# Chip-to-chip link traffic (on systems that have the link)
--section C2CLink
```

---

## Profiling multiple kernel launches

If the kernel is called multiple times and you want different iterations:

```bash
# Skip the first 5 invocations, then profile 3 consecutive launches
ncu -k "regex:my_kernel" -s 5 -c 3 -o report.ncu-rep ./harness
```

`-s N` skips the first N matches. `-c N` limits to N matches. Useful for ignoring warmup or focusing on a steady-state iteration. For a program that warms up first (JIT, autotuning), `--profile-from-start off` plus a profiler range around the call you want (`cudaProfilerStart/Stop`, `torch.cuda.profiler.start()/stop()`) profiles only that call.

A report with several launches holds one action per launch; the helpers analyze each (tags `<tag>.<i>_<function>`), and `--kernel REGEX` picks some.

---

## GPU clocks and measurement conditions

Record the collector's actual clock-control default or an explicitly supported override, plus observed frequency. Jetson power/clock controls are platform-specific; preflight proposes `--clock-control none` for identified Orin/Thor targets or an observed L4T platform and changes no settings. Read the selected target guide before changing clocks. Use the same clock, power and thermal conditions for an A/B comparison.

`sm__cycles_elapsed.avg.per_second`, when collected, records the observed SM clock. NCU 2026.1/2026.3 B200/B300 defaults and examples are in [the version-specific notes](architectures/sm100/collection.md#gpu-clocks). Do not copy a desktop `nvidia-smi` clock lock to Jetson.

## Gotchas

- **Profile run time blows up with `--set full`**: that's normal, each replay is a full kernel execution.
- **`regex:...` doesn't match anything**: `-k` matches the kernel's function name by default (`--kernel-name-base function`): `my_kernel` for `void my_kernel<(int)8, (int)256>(float *)`. To tell template instances apart, add `--kernel-name-base demangled` and match the full name. List the names with the survey (Recipe 0) or `cuobjdump -res-usage ./harness | grep Function | cu++filt`.
- **No report file**: ncu writes none when it profiled nothing (`==WARNING== No kernels were profiled.`): the `-k` regex didn't match, `-c`/`--launch-skip` skipped past the launches, or the program exited before the kernel ran. ncu does not overwrite an existing report either (`==ERROR== File ... already exists`): pass `-f`, or give each run its own name.
- **PM Sampling returns nothing**: check the installed collector/target supports it and that it was collected. `plot_timeline.py --list` shows report coverage. No timeline means temporal conclusions are unavailable; it does not turn static measurements into zero.
- **"Could not deploy stock section files to $HOME"**: make the configured cache location writable using the supported collector configuration; do not silently repurpose the shell's home directory.
