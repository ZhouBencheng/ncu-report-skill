# B200/B300 collector-version details

These are retained observations for the stated B200/B300 and NCU versions. The shared [collection workflow](../../03-collection.md) owns launch selection, survey, source import, warm-up, replay and report export. Use the installed `ncu --help`, `--list-sets` and `--list-sections` before applying these details on another version.

## What each `--set` contains

```bash
ncu --list-sets           # list all sets
ncu --list-sections       # list all sections
```

ncu 2026.3:

| Set | Sections included | Use when |
|---|---|---|
| `basic` (default) | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution | Smoke test — is ncu working at all? |
| `detailed` | basic + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis (+ chart), SourceCounters, SpeedOfLight_RooflineChart, Tile | Middle ground |
| `full` | everything above + InstructionStats, MemoryWorkloadAnalysis_Tables, NumaAffinity, Nvlink_Tables / _Topology, PmSampling, SchedulerStats, the hierarchical roofline charts, WarpStateStats, gr10x_UTCMMA_Inference / _Training | Broad overview when this version and hardware support the sections. |
| `pmsampling` | PmSampling, gr10x_UTCMMA_Inference, gr10x_UTCMMA_Training | Timelines only (the PM series and the timed warp samples) |
| `roofline` | SpeedOfLight + the roofline charts, WorkloadDistribution | Roofline only |
| `nvlink` | Nvlink, Nvlink_Tables, Nvlink_Topology | Multi-GPU traffic |

ncu 2026.3 removed the `PmSampling_WarpStates` section of 2026.1 (`--section PmSampling_WarpStates` is an error there); the stall reasons over time come from the timed warp samples of PmSampling instead (`plot_timeline.py`). With ncu 2026.1, add `--section PmSampling_WarpStates` for its stall-reason PM series.

---

## GPU clocks

ncu fixes the clock itself: it locks the GPU clock for every profiled kernel and restores it after, so durations and
throughputs are comparable between runs. The default is `--clock-control boost` in ncu 2026.1–2026.3 and `base` in older
versions (`ncu --help` shows it); on a B300, `base` (~1.06 GHz) made kernels ~1.7x slower than `boost` (~2.0 GHz).
Name the setting in the command so the report says what it ran at. No outer lock is needed.

To profile at a fixed clock of your own (to match the clock a timing A/B ran at), lock it around the command and pass
`--clock-control none`; without it, ncu replaces your lock with its own:

```bash
# or lock it the way your system provides; check with nvidia-smi -q -d CLOCK
sudo nvidia-smi -lgc <mhz>,<mhz>
ncu --clock-control none --set full ... ./harness [args]
sudo nvidia-smi -rgc
```

Read the clock a report ran at from `sm__cycles_elapsed.avg.per_second` (in `metrics_key_<tag>.txt`), and give it in the report.

---
