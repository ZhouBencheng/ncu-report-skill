# `ncu_report` Python API

Use the Python module, not CLI output, for anything beyond a quick look. The API lets you extract, aggregate, compare, and archive metric data cleanly.

> **Jetson AGX Orin (sm_87):** read Orin reports with the aarch64 `ncu-report==2026.3.1` wheel (a reader distinct from the on-device NCU 2025.2.1 collector) and the `--profile orin` curated view. The exact reader invocation, `NCU_REPORT_PYTHONPATH` bridging, and fail-loud behaviour are in [`10-orin-collector-reader.md`](10-orin-collector-reader.md).

Module path (adjust for your CUDA version):
```bash
export PYTHONPATH=$PYTHONPATH:/opt/nvidia/nsight-compute/2026.1.1/extras/python
python3 -c "import ncu_report; print('OK')"
```

Standalone NCU installations can instead use `/opt/nvidia/nsight-compute/<version>/extras/python`; the helpers probe this layout alongside the existing CUDA paths. The inspected NCU 2026.1.1.0 module was importable at the following path (verify the tool version with `ncu --version`, not its directory name):

```bash
PYTHONPATH=/opt/nvidia/nsight-compute/2026.1.1/extras/python${PYTHONPATH:+:$PYTHONPATH} \
    python3 -c 'import ncu_report; print(ncu_report.__file__)'
```

---

## Basic loading

For analysis, load through the detached capture (`ncu_utils`): each report is read
once in a fresh child process — the only importer of `ncu_report` — and every
range/action is enumerated. `select_actions` returns plain dicts, so the parent
holds no native handle and an auxiliary launch is never dropped.

```python
import ncu_utils as u

capture = u.load_capture("path/to/full_<tag>.ncu-rep")  # one fresh child per report
for act in u.select_actions(capture):                   # ALL actions, not index 0
    print(u.action_name(act), "-", len(act.get("metrics") or {}), "metrics")

# Select one action explicitly; an unknown/ambiguous selector raises rather than
# silently returning action 0.
only = u.select_actions(capture, "r0a1")[0]
```

Under the hood the capture child wraps the native primitives below; reach for them
only when extending the capture itself. A report can contain multiple "ranges"
(each range = one profiled region); with `-c 1` you have one range containing one
action. Never take `action_by_idx(0)` as *the* action — that drops every auxiliary
launch in a multi-action report.

```python
import sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
import ncu_report

report = ncu_report.load_report("path/to/full_<tag>.ncu-rep")
rng = report.range_by_idx(0)
action = rng.action_by_idx(0)   # ONE action; iterate range_by_idx/action_by_idx for all

print(f"Kernel demangled name: {action.name()}")       # void my_kernel<8, 256>(...)
print(f"Total metrics collected: {len(action.metric_names())}")
```

The native-API sections that follow operate on a native `action`/`metric` handle
like this (what the capture child wraps), distinct from the plain-dict `action`
that `select_actions` returns for analysis.

---

## Reading a single metric

Do **not** reach into the native `ncu_report` objects directly, and do **not**
collapse a read error into a default with a bare `try/except`. Load a detached
capture once (fresh child process) and read through `ncu_utils`, which keeps the
five value states — `absent`, `read-error`, `nan`, `+inf`/`-inf`, `true-zero`,
and an ordinary `value` — distinguishable.

```python
import ncu_utils as u

capture = u.load_capture("profile.ncu-rep")   # one fresh child per report
# "A single metric" still means ONE explicitly chosen action — not a silent [0].
# Pass an explicit selector; iterate u.select_actions(capture) to see them all.
action = u.select_actions(capture, "r0a0")[0]

# classify_value takes the detached VALUE (what metric_value returns), NOT the
# whole record. metric_value returns a number for 'value'/'true-zero', None when
# absent, and an explicit sentinel dict for read-error / NaN / ±inf — never a
# silent default. classify_value names which state the value is in.
for name in ["sm__throughput.avg.pct_of_peak_sustained_elapsed",
             "dram__bytes_read.sum.per_second",
             "gpu__time_duration.sum"]:
    state = u.classify_value(u.metric_value(action, name))
    print(f"{name}: {u.metric_display(action, name)}  ({state})")
```

**Never substitute a default for a failed or absent read, and gate arithmetic on
availability as well as a numeric state.** `metric_display` already combines both
axes: it returns the number only when the scalar is present, readable, *available*
(`has_value`), and numeric — otherwise `<absent>`, `<read-error>`, `<unavailable>`,
or `<nan>`/`<+inf>`. A true-zero whose `has_value` is false must not be divided as
if it were a measurement, so guard on the rendered value's type:

```python
bw = u.metric_display(action, "dram__bytes_read.sum.per_second")
if isinstance(bw, (int, float)):          # numeric AND available
    print(f"DRAM read BW: {bw/1e9:.2f} GB/s")
else:
    print(f"DRAM read BW: {bw} (not a measured value)")
```

Metric names differ between GPU generations — see
[`08-b200-metric-names.md`](08-b200-metric-names.md). In the inspected Thor
reports, curated metrics such as `dram__bytes_read.sum` and
`dram__bytes_read.sum.per_second` were **absent**; the capture records them as
`absent`, not as `0`. The B200 curated list remains unchanged.

---

## Enumerating available metrics

```python
# Full list — 2000+ metrics for --set full
all_names = action.metric_names()

# Filter by pattern
for name in sorted(all_names):
    if "warps_issue_stalled" in name and "ratio" in name:
        try:
            print(name, "=", action[name].value())
        except Exception as e:
            print(name, "= <error>", e)   # report the failure; never a silent default
```

This is how you discover the *actual* metric names available on your GPU instead of guessing.

---

## Per-instance (per-SM, per-PC, per-time-sample) values

Many metrics have multiple values per collection. `value()` returns the aggregate (sum / avg depending on `rollup_operation`), but you can also enumerate the individual samples.

```python
m = action["pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg"]
n = m.num_instances()              # e.g., 1660 for a PM-sampled metric
print(f"instances: {n}")

vals = []
for i in range(n):
    try:
        v = m.as_double(i)
    except Exception:
        try:
            v = float(m.as_uint64(i))
        except Exception:
            v = None
    vals.append(v)
```

For PM sampling this is the timeline — index `i` is a time-ordered sample. Bucket these for an ASCII plot (see [`../scripts/plot_timeline.py`](../scripts/plot_timeline.py)).

The inspected Thor `sm_110` and `sm_110a` reports each contained 15 PM metrics with 252–541 samples per metric and no unreadable samples. None of the ten default plotter metrics appeared, so their `no instances` output describes those requested metrics, not PM sampling as a whole. Select an exact name from `action.metric_names()` using the existing CLI:

```bash
python3 scripts/plot_timeline.py --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag '<tag>' \
    --metric pmsampling:sm__inst_executed_pipe_alu_realtime.avg.pct_of_peak_sustained_elapsed \
    --metric pmsampling:sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed \
    --cols 600
```

For these reports, `--cols 600` avoids bucket reduction of the active samples; the plotter trims leading and trailing measured zeros. The tensor metric's samples were all measured zero for this harness. This does not establish tensor-instruction support or activity in other workloads. Missing or unreadable samples must not be interpreted as zero; the inspected PM series did not exercise that case. For other reports, use `--cols` no smaller than the largest selected series' instance count; the compressed path can omit trailing remainder samples and is outside this validation.

---

## Per-PC → per-source-line mapping

For a report containing per-PC samples collected with `--section SourceCounters` (and `--set source` only where listed), map them to source lines via `action.source_info(pc)`. Check correlation data in the report; section availability alone is insufficient:

```python
def per_pc_stalls(action, stall_metric):
    m = action[stall_metric]
    n = m.num_instances()
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        pc = cor.as_uint64(i)
        val = m.as_uint64(i)
        si = action.source_info(pc)
        if si is None:
            file, line = "?", 0
        else:
            file, line = si.file_name(), si.line()
        out.append((file, line, val))
    return out

stalls = per_pc_stalls(action, "smsp__pcsamp_warps_issue_stalled_long_scoreboard")
```

Aggregate by `(file, line)` and sort by total to get hottest stall lines. See [`../scripts/extract_stall_hotspots.py`](../scripts/extract_stall_hotspots.py) for a complete implementation.

---

## Discovering Value Kind

Each metric has a value kind (uint64, double, float, string). Use `m.kind()` to check before calling the right accessor:

```python
def metric_val(m, i=None):
    k = m.kind()
    VK = m.ValueKind_UINT64, m.ValueKind_DOUBLE, m.ValueKind_FLOAT, m.ValueKind_STRING
    if i is None:
        return m.value()          # aggregate
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    # Try generic conversions as fallbacks
    try:
        return m.as_uint64(i)
    except Exception:
        return m.as_double(i)
```

---

## Useful `action` / `metric` methods

```python
# Action (= one kernel launch's profile data)
action.name()                   # demangled kernel name
action.metric_names()           # list of all metrics
action.metric_by_name(name)     # same as action[name]
action.source_info(pc)          # IPC → SourceInfo (file, line) — only if -lineinfo was used
action.sass_by_pc()             # dict {pc → SASS instruction string}
action.ptx_by_pc()              # dict {pc → PTX} — only if --keep was used
action.rule_results_as_dicts()  # NCU rule-engine output as list of dicts

# Metric
m.value()                       # aggregate value
m.unit()                        # string, e.g. "%" or "cycle"
m.kind()                        # value kind (UINT64 / DOUBLE / ...)
m.rollup_operation()            # AVG / MAX / MIN / SUM / NONE
m.num_instances()               # per-instance count (0 if aggregate only)
m.has_correlation_ids()         # True for per-PC metrics
m.correlation_ids()             # parallel array for num_instances
m.description()                 # human-readable metric description
```

---

## Exploring when you don't know the right metric name

```python
# Print all metric names sorted
for n in sorted(action.metric_names()):
    print(n)

# Print metrics matching a pattern with their current value
import re
pat = re.compile(r"dram__bytes.*sum")
for n in sorted(action.metric_names()):
    if pat.search(n):
        try:
            v = action[n].value()
            print(f"{n} = {v}  (unit: {action[n].unit()})")
        except Exception as e:
            print(f"{n} = ERROR {e}")
```

This is how I built [`08-b200-metric-names.md`](08-b200-metric-names.md) — by enumerating everything available on sm_100.

---

## Comparing two reports programmatically

Load each report in its **own** fresh capture (never two live reports in one
process), enumerate actions, and render with `metric_display` so an absent /
read-errored / unavailable metric stays distinct from a measured value — and a
true-zero baseline is never dropped from the comparison. This mirrors what
[`../scripts/analyze_reports.py`](../scripts/analyze_reports.py) already does.

```python
import ncu_utils as u

def compare(report_paths_by_tag, metrics, selector=None):
    # One fresh child per report; the parent keeps only the detached dicts.
    # Enumerate EVERY action (keyed by tag + action label) so an auxiliary launch
    # is never silently dropped. Pass an explicit `selector` (e.g. "r0a1") only to
    # deliberately narrow to one action — a bare [0] would discard the rest.
    columns = []   # (column_label, action)
    for tag, path in report_paths_by_tag.items():
        capture = u.load_capture(path)
        for action in u.select_actions(capture, selector):
            columns.append((f"{tag}/{u.action_label(action)}", action))
    print(f"{'Metric':<75}" + "".join(f"{c:>20}" for c, _ in columns))
    for m in metrics:
        row = f"{m:<75}"
        for _, action in columns:
            # number, <absent>, <read-error>, <unavailable>, or nan/+inf — distinct
            v = u.metric_display(action, m)
            v = f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
            row += f"{v:>20}"
        print(row)

compare({"v1": "v1.ncu-rep", "v2": "v2.ncu-rep"}, [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "l1tex__t_sector_hit_rate.pct",
])
```

A percent-change column is only meaningful between two *numeric, available*
readings. Guard both sides with `isinstance(u.metric_display(a, m), (int, float))`
(which checks numeric state **and** `has_value` availability) before dividing;
an unavailable raw zero is not a measurement. A true-zero baseline makes the ratio
undefined (not skippable) — report it as such rather than hiding the row, and
never collapse an absent reading into `0`.

---

## Extracting NCU's rule suggestions

The rule engine results — the "OPT Est. Speedup: X%" bullets — are accessible as structured data:

```python
for result in action.rule_results_as_dicts():
    severity = result.get("type", "?")      # "OPT" / "INF" / "WRN"
    rule_name = result.get("rule_name", "?")
    message = result.get("message_for_display", "?")
    est_speedup = result.get("estimated_speedup_pct", None)
    print(f"[{severity}] {rule_name}: {est_speedup}%")
    print(f"    {message[:200]}")
```

Sort by `estimated_speedup_pct` descending to get the highest-impact suggestions first.

---

## Saving everything for later

The CLIs already write the **lossless** archive for you: `analyze_reports.py` emits
`analysis/metrics_all_<slug>.json` — one entry per metric carrying the typed value,
unit, value-kind, per-instance values/kinds, correlation IDs/units, the scalar
`has_value`, and any per-read error, with explicit sentinels for
absent / read-error / NaN / ±inf / true-zero. Re-analysis reads that JSON instead
of reopening the `.ncu-rep`:

```python
import json
from pathlib import Path

metrics = json.loads(Path("analysis/metrics_all_<slug>.json").read_text())
rec = metrics["gpu__time_duration.sum"]
print(rec["value"], rec.get("unit"), rec.get("has_value"))
```

Do **not** hand-roll a scalar-only dump (`{name, value, unit}` from `m.value()`):
it discards instances, IDs, correlation IDs/units, value-kind, `has_value`, and
read errors, and a naive `json.dump` of a NaN/±inf value is not even valid JSON.
The capture child writes the record under `json.dump(..., allow_nan=False)` with
the sentinels intact — archive *that*, never a lossy re-derivation.

---

## Gotchas

- **`KeyError` on a metric that "should" exist**: the metric has a different name on this GPU. Check [`08-b200-metric-names.md`](08-b200-metric-names.md) or enumerate with `action.metric_names()`.
- **`num_instances() == 0`** but you expected per-instance data: the metric wasn't collected in instanced mode, or the section that produces it wasn't requested. Re-run ncu with the right `--section`.
- **`has_correlation_ids() == False`** on a source-level metric: `-lineinfo` wasn't on the compile line. Rebuild.
- **`source_info(pc)` returns None**: same as above — rebuild with `-lineinfo`.
- **Metric value is a string** like `"PolicySpread"`: it's an enum, use `m.as_string()` or `m.value()` and expect a string.
