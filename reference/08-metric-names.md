# Metric availability and interpretation

Run `helpers/preflight.py` on the collection machine to record the installed collector, available sets/sections and metric inventory. When reading a report, `action.metric_names()` is authoritative about what was captured. A supported metric may still be absent from a particular report.

`helpers/analyze_reports.py` selects candidate metrics from the report's device identity. `metrics_evidence_<tag>.json` preserves each raw name, unit, value and state:

- `present`: includes a measured zero.
- `not_in_report`: unsupported **or** uncollected; the report alone cannot distinguish them.
- `scalar_unavailable`: the field has no readable scalar (it may still hold instances).
- `read_error`: the reader could not access the value.

Do not fill missing values with zero. A preflight `not_listed` result describes that collector/device query, not all versions of that architecture. Check the report's own collector metadata before using an inventory from another run.

Candidate metric names are not semantic aliases: retain `.ratio` versus `.pct`, `active` versus `elapsed`, sums versus averages, and actual units. Never subtract or compare unlike normalizations. Cross-device utilization percentages refer to different device peaks; they are not a speedup measurement.

For the existing B200/B300 names and NCU 2026.1/2026.3 behavior, read [SM100 metric reference](architectures/sm100/metric-reference.md). Other targets start with their selected architecture guide.

Official references: [NCU metrics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide), [CLI metric queries](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
