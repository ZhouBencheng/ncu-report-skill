---
name: ncu-report-skill
description: Profile CUDA kernels and interpret Nsight Compute reports using the actual GPU and collector environment. Use for measured bottleneck diagnosis, NCU reports, source stalls, timelines, and optimization plans on B200/B300, RTX 4090, Jetson Orin/Thor, or other supported NVIDIA GPUs.
---

# CUDA kernel profiling

Profile → diagnose → plan. Rank recommendations by measured evidence and state what was unavailable. A candidate metric, a successful parse, and a performance improvement are different claims.

## Select the environment before loading architecture advice

Before collection, choose a new run directory following [the project artifact convention](reference/00-directory-layout.md), create it, and set `PROFILE_RUN_DIR` to its path.

1. Identify the **runtime GPU and CC**, and separately record the kernel's **compiled target**, CUDA/driver, collector, and report-reader versions. An `a`/`f` compiler target is not a runtime CC. Unknown devices receive generic analysis; never default to B200.
2. For collection, run `python3 helpers/preflight.py --ncu /actual/path/to/ncu --output "$PROFILE_RUN_DIR/preflight.json"` on the GPU machine. Read [collection](reference/03-collection.md), then only the selected target guide. Discovery does not prove counter permission or successful collection.
3. For an existing report, `helpers/analyze_reports.py` reads its own device identity. `--target` accepts an exact SM or listed product alias; conflicting identity fails, missing identity remains explicit-unverified.

| Runtime target | Read when selected |
|---|---|
| SM87 / Orin | [Ampere Jetson](reference/architectures/sm87/guide.md) |
| SM89 / RTX 4090 | [Ada](reference/architectures/sm89/guide.md) |
| SM90 / H100/H200 | [Hopper](reference/architectures/sm90/guide.md) |
| SM100 / B200 | [B200](reference/architectures/sm100/guide.md) |
| SM103 / B300 | [B300](reference/architectures/sm103/guide.md) |
| SM110 / Thor | [Thor](reference/architectures/sm110/guide.md) |

Thor is CC 11.0, not SM120. The Blackwell family name alone does not select a capability set. Load common references plus one target guide; follow its detailed links only when needed.

## Shared workflow

1. Use a new run directory for each experiment, following the project's artifact convention. Keep `harness/`, `reports/`, `analysis/` and `REPORT.md` together ([layout](reference/00-directory-layout.md)). Select representative shapes and record the question being measured.
2. Profile the real code if it already has line info; otherwise use a bounded harness with `-lineinfo` ([harness guide](reference/02-harness-guide.md)). Choose the compile target from the kernel's required instructions and installed compiler, not from a copied B200 command.
3. Collect with the installed, supported recipe ([collection](reference/03-collection.md)). Preserve the command, environment and raw report. PM sampling and source data are conditional; their absence does not invalidate available static measurements.
4. Analyze with the shared helpers:

   ```bash
   python3 helpers/analyze_reports.py --run-dir "$PROFILE_RUN_DIR" \
       --report "$PROFILE_RUN_DIR/reports/full_case.ncu-rep" --tag case
   python3 helpers/extract_stall_hotspots.py --run-dir "$PROFILE_RUN_DIR" \
       --report "$PROFILE_RUN_DIR/reports/full_case.ncu-rep" --tag case
   python3 helpers/plot_timeline.py --run-dir "$PROFILE_RUN_DIR" \
       --report "$PROFILE_RUN_DIR/reports/full_case.ncu-rep" --tag case
   ```

   Run source/timeline helpers when those measurements exist. For inlined library code use the hotspot helper's `--binary` on the exact profiled binary. [Helper details](helpers/README.md).
5. Evaluate [analysis dimensions](reference/05-analysis-dimensions.md) and [diagnosis patterns](reference/06-diagnosis-playbook.md). Match the proposed implementation to the selected target's supported capabilities. Compare like units, aggregation and active/elapsed denominators.
6. Write an [evidence-backed report](reference/07-report-template.md). Cite actual metric values and states, source lines, workload, environment, and the limitations of each conclusion. Verify any improvement with a controlled timing comparison.

`metrics_evidence_<tag>.json` records target selection, separate collector/reader provenance and metric states. Supply the recorded `--collector-version` once per report when known; otherwise the producer version remains unknown. **Missing is not zero:** report absence cannot by itself distinguish unsupported from uncollected. The full raw-metric archive remains available even when the curated target view omits a field. [Metric semantics](reference/08-metric-names.md).

For permission, reader or sampling failures, read [common issues](reference/09-common-issues.md). The [Python API reference](reference/04-python-api.md) and [workflow checklist](reference/01-workflow.md) are supporting details, not mandatory additional context on every request.
