# Validation evidence — 2026-10-27

These records describe the actual tested devices and installed toolchains, not universal architecture constants. No clock/power settings were changed and no optimization speedup is claimed.

| Target | CUDA | Collector | Reader | Recorded configuration |
|---|---|---|---|---|
| Orin SM87 | 12.6.85 | 2025.2.1 | 2026.1.1, separate aarch64 host | L4T R36.4.7, 14 SM, MAXN |
| RTX 4090 SM89 | 12.4.131 | 2024.1.1 | 2024.1.1, same host | Ubuntu 22.04.3, 128 SM |
| Thor SM110 | 13.2.86 | 2026.1.1 | 2026.1.1, same host | L4T R39.2.1, 20 SM, MAXN |

The supplied Orin installation's bundled `_ncu_report.so` was x86-64 and could not load on its aarch64 host. The installed aarch64 2026.1.1 reader on the validation host read its 2025.2.1 reports. This is an observed packaging limitation, not a claim about every 2025.2.1 package. No packages or system tools were installed for this validation.

The 4090's 2024.1.1 reader has no `rule_results` or `timed_warp_samples` API. The analyzer reports rule extraction as unavailable; scalar metrics, source counters and PM series still work. The collector's CLI details page remains an alternative source of rules.

## What ran

Task-authored [CUDA correctness probe](../tests/gpu/validation.cu) and [instruction probe](../tests/gpu/instruction-probe.cu) compiled natively on all three targets. They checked vector copy, block reduction, FP32 64×64×64 SIMT matmul, one ordinary 4-byte `cp.async` copy case and one BF16-input/FP32-accumulator WMMA 16×16×16 case. These are bounded correctness probes, not an optimized benchmark suite or validation of TMA/tcgen05/TMEM/2-CTA kernels.

Each target collected `basic` reports for three kernels and `basic + SourceCounters + PmSampling` for one `vector_copy` launch, using `--clock-control none`. Counter access used the machine's available `sudo -n` on the two Jetsons and the ordinary user on 4090. The real report reader extracted scalar fields and confirmed nonempty source-counter/PM data. All three helper CLIs (analyzer, source hotspots, timeline listing/rendering) were executed against these reports with the reader pairings above.

Receipts retain exact build/collection commands, report hashes, source hashes, observed device properties and limits:

- [SM87 receipt](architectures/sm87/validation.json)
- [SM89 receipt](architectures/sm89/validation.json)
- [SM110 receipt](architectures/sm110/validation.json)

The [helper validation receipt](helper-validation.json) records the tested helper/profile hashes, collector/reader provenance and per-target CLI results.

Binary reports are experiment artifacts and are excluded from the skill package. A small extracted scalar fixture tests routing/value preservation without a GPU. Reproduce from the committed probe source and receipt commands, using a new experiment directory and the actual machine's permitted counter access.

## Regression scope

`python3 -m unittest discover -s tests -v` checks the original B200/B300 metric names/order and value preservation, identity conflicts, unknown-target fallback, metric states/units, version-dependent API absence, and missing timeline samples. New probe coverage does not imply every technique, dtype or shape was validated.

B200, B300 and Hopper were not available for new hardware runs. Their retained references and B200/B300 metric behavior have offline coverage only. The baseline B200 knowledge remains available by target; newly added device knowledge is not used to rewrite its benchmark conclusions.
