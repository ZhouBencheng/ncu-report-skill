# Validation evidence — 2026-10-27

These results describe the tested devices and installed toolchains, not universal architecture constants. No clock/power settings were changed, no packages or system tools were installed, and no optimization speedup is claimed.

## Observed environments

| Target | CUDA / nvcc | NCU collector | NCU Python reader | Native compile target |
|---|---|---|---|---|
| Orin SM87 | 12.6.85 | 2025.2.1 | 2026.1.1, separate aarch64 validation host | `sm_87` |
| RTX 4090 SM89 | 12.4.131 | 2024.1.1 | 2024.1.1, same host | `sm_89` |
| Thor SM110 | 13.2.86 | 2026.1.1 | 2026.1.1, same host | `sm_110` |

| Target | OS / platform | Driver | Observed power mode | Counter access |
|---|---|---|---|---|
| Orin | Ubuntu 22.04.5 LTS / L4T R36.4.7 | 540.4.0 | MAXN | Available `sudo -n` |
| RTX 4090 | Ubuntu 22.04.3 LTS | 550.163.01 | Not recorded | Ordinary user |
| Thor | Ubuntu 24.04.4 LTS / L4T R39.2.1 | 595.78 | MAXN | Available `sudo -n` |

The CUDA runtime reported the following resources for those configurations. Shared-memory sizes are in KiB; L2 sizes are in MiB.

| Target | SMs | CUDA-visible memory (bytes) | L2 | Shared memory / block, opt-in | Shared memory / SM | Max blocks / SM |
|---|---|---|---|---|---|---|
| Orin | 14 | 32,188,633,088 | 4 | 163 | 164 | 16 |
| RTX 4090 | 128 | 25,386,352,640 | 72 | 99 | 100 | 24 |
| Thor | 20 | 131,920,637,952 | 32 | 227 | 228 | 24 |

All three reported 48 KiB default shared memory per block, 65,536 registers per SM, 32-thread warps, a maximum of 1,536 threads per SM and 1,024 threads per block. Orin and Thor reported integrated GPUs; the 4090 did not. These observed values do not establish limits for every SKU of an architecture.

## Bounded correctness probes

The two task-authored CUDA probe sources compiled natively on all three targets with `-O2 -lineinfo` and the compile targets above. Both executables completed within a 15-second timeout on each device. Every case below passed on Orin, RTX 4090 and Thor.

| Probe | Workload | Observed result |
|---|---|---|
| Vector copy | 262,144 elements | Correct |
| Block reduction | 262,144 elements | Correct |
| SIMT matrix multiply | FP32, 64 × 64 × 64 | Correct; maximum absolute error 0 |
| Ordinary `cp.async` global-to-shared copy | 4-byte copies, 65,536 elements | Correct |
| One-warp WMMA | BF16 inputs, FP32 accumulation, 16 × 16 × 16 | Correct; maximum absolute error 0 at absolute tolerance 1e-5 |

These are bounded correctness checks, not an optimized benchmark suite. They do not validate other shapes, dtypes or specialized instruction paths.

## Collection and report reading

Each target collected one `basic` report containing three kernel launches (copy, reduction and SIMT matmul), plus one `basic + SourceCounters + PmSampling` report restricted to one `vector_copy` launch. All collections used `--clock-control none` and `--import-source yes`; both collection and scalar extraction passed with the collector/reader pairings above.

| Target | Basic metrics per kernel | Detail metrics | Nonempty PC-sampling metrics in detail | PM series / recorded instances |
|---|---|---|---|---|
| Orin | 501 | 618 | 37 | 1 series, 414 instances |
| RTX 4090 | 491 | 606 | 37 | 1 series, 22 instances |
| Thor | 561 | 708 | 53 | 15 series, 64–290 instances per series |

The basic reports contained no PM series or nonempty PC-sampling metrics. Source-counter and PM evidence comes from the detail reports; a successful basic parse alone does not establish that coverage.

All three report helpers ran against the collected reports with automatic target selection. The analyzer processed all four launches per target; source-hotspot extraction and timeline listing/rendering used the detail report. Each CLI exited 0. Separate collector/reader provenance was checked in all four analysis outputs per target. Read-only preflight also passed all five collector queries (version, sets, sections, help and metrics):

| Target | Detected target | Discovered metric names | Proposed collection options |
|---|---|---|---|
| Orin | `sm87` | 122,862 | `--set full --import-source yes --clock-control none` |
| RTX 4090 | `sm89` | 118,573 | `--set full --import-source yes` |
| Thor | `sm110` | 139,085 | `--set full --import-source yes --clock-control none` |

The Jetson preflights observed L4T and used a task-local NCU-only wrapper for available `sudo -n` access. The 4090 used ordinary user access. Preflight proposed options; the actual report collections used the bounded recipes above, not the full set. Metric inventory sizes describe the installed collector queries, not fields captured in every report.

## Reader limitations

The supplied Orin installation's bundled `_ncu_report.so` was x86-64 and could not load on its aarch64 host. The installed aarch64 2026.1.1 reader on the separate validation host read its 2025.2.1 reports. This is an observed packaging limitation, not a claim about every 2025.2.1 package.

The 4090's 2024.1.1 reader has no `rule_results` or `timed_warp_samples` API. The helpers reported those APIs as unavailable; scalar metrics, source counters and PM series still worked. The collector's CLI details page remains an alternative source of rules. The 2026.1.1 readers exposed `rule_results`, `timed_warp_samples` and `source_info`; API presence alone does not establish data coverage for every report.

## Provenance and scope

The measured results, exact commands, full source/report hashes and helper/profile versions are preserved in Git commit [`c0a4947`](https://github.com/ZhouBencheng/ncu-report-skill/tree/c0a494779bc7d65ea5ab4d8c1f80cb8220b19f4a). Its historical [CUDA correctness probe](https://github.com/ZhouBencheng/ncu-report-skill/blob/c0a494779bc7d65ea5ab4d8c1f80cb8220b19f4a/tests/gpu/validation.cu), [instruction probe](https://github.com/ZhouBencheng/ncu-report-skill/blob/c0a494779bc7d65ea5ab4d8c1f80cb8220b19f4a/tests/gpu/instruction-probe.cu) and validation receipts retain the original evidence. Probe sources, regression fixtures and experiment receipts are excluded from the installed skill package. Binary reports are experiment artifacts and are also excluded.

For identifying the measured reports, these are the first 12 hexadecimal characters of each SHA-256; full hashes remain in the historical receipts.

| Target | Basic report | Detail report |
|---|---|---|
| Orin | `40044ccb7506` | `77c45e4f0e14` |
| RTX 4090 | `0748a1e7fe8f` | `4c8fd3f469a9` |
| Thor | `5bf025afd00f` | `7cdee8f02631` |

The offline regression sources at that commit cover original B200/B300 metric names/order and value preservation, identity conflicts, unknown-target fallback, metric states/units, version-dependent API absence and missing timeline samples. B200, B300 and Hopper were not available for new hardware runs; their retained references and B200/B300 metric behavior have offline coverage only. The B200 benchmark conclusions remain scoped to their original setup.

TMA, WGMMA, `tcgen05`, TMEM, 2-CTA cooperative kernels, other shapes/dtypes and optimization speedups remain unproven by this validation. A successful parse or correctness probe does not establish performance.

New profiling runs keep harnesses, reports, preflight JSON and analysis JSON under the profiled project's [run directory](00-directory-layout.md), outside the skill package. The eight shipped JSON files are runtime architecture profiles and metric candidate data used by the helpers.
