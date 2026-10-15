#!/usr/bin/env python3
"""Detached NCU capture: read one report in a fresh process, emit typed JSON.

This module is the ONLY importer of the native ``ncu_report`` module, and it
imports it lazily (inside ``capture``), so the rest of the skill — and every CPU
test — imports and runs without the native SDK present. Run it once per report
in a fresh interpreter; the parent process consumes the detached JSON record and
never touches a live native object.

The schema mirrors the independently validated oracle
(``native_ncu_capture.py``, schema 3): missing metrics, native read errors and
non-finite values stay distinct from numeric zero, instance kind/value are
archived without ever calling the non-existent ``has_value(index)`` overload,
and PC samples are aggregated without zero-fill, top-k truncation, or mixing the
not-issued subset into the totals.

CLI:
    python3 -I ncu_capture.py --report R.ncu-rep --output capture.json

PYTHONPATH (or the NCU_REPORT_PYTHONPATH env var) must point at the selected
report READER's module directory so the child can import ``ncu_report``. The
reader is separate from the collector: on the Orin controller the reader is the
aarch64 ``ncu-report==2026.3.1`` wheel reading a report COLLECTED by Orin's NCU
2025.2.1 (a newer reader is not a newer collector); on Thor the reader is NCU
2026.1.1's ``extras/python``. A missing native reader fails loudly here — there
is no fake/fallback reader in the production path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = 3

# Sentinel key used to encode a non-finite float (NaN / +inf / -inf) in JSON,
# which has no native representation for them. Kept distinct from numeric zero
# and from a native read error.
NONFINITE_KEY = "nonfinite"
NATIVE_ERROR_KEY = "native_error"


def sdk_path_entries(raw):
    """Split a PYTHONPATH-style value into safe sys.path entries.

    Returns only absolute, existing directories. Empty and relative entries are
    dropped: under ``python -I`` the current directory is deliberately off the
    import path, and inserting a relative (or empty, which means cwd) entry would
    silently re-enable it — exactly the planted-module hazard isolation prevents.
    """
    out = []
    for part in (raw or "").split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        p = Path(part)
        if p.is_absolute() and p.is_dir():
            out.append(str(p))
    return out


def insert_sdk_path(raw):
    """Prepend validated SDK directories to sys.path; return what was inserted."""
    inserted = []
    for entry in sdk_path_entries(raw):
        if entry not in sys.path:
            sys.path.insert(0, entry)
            inserted.append(entry)
    return inserted


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def detached(value):
    """Convert a native value into a JSON-safe, round-trippable form.

    A non-finite float becomes ``{"nonfinite": "nan"|"inf"|"-inf"}`` so it stays
    distinct from numeric zero and survives ``json.dump(..., allow_nan=False)``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return {NONFINITE_KEY: repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [detached(item) for item in value]
    if isinstance(value, dict):
        return {str(key): detached(item) for key, item in value.items()}
    return {"unsupported_native_type": type(value).__name__, "repr": repr(value)}


def error_record(error):
    return {NATIVE_ERROR_KEY: {"type": type(error).__name__, "message": str(error)}}


def call(obj, method, *args):
    """Call a native accessor, returning a detached value or an error record."""
    try:
        return detached(getattr(obj, method)(*args))
    except Exception as error:  # native bindings raise bare Exception subclasses
        return error_record(error)


def is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def value_record(metric):
    """Archive native observations without inventing instance availability.

    The installed native ``IMetric`` exposes ``has_value()`` and
    ``has_value(ValueKind)`` only — there is no instance-index overload — so the
    scalar ``has_value()`` is kept separate from the per-instance kind/value
    arrays, which are read via ``value(index)``/``kind(index)`` directly.
    ``has_value(index)`` is NEVER called.
    """
    record = {field: call(metric, field) for field in
              ("value", "has_value", "kind", "unit", "num_instances")}
    count = record["num_instances"]
    record["native_instance_availability"] = "not_exposed_by_has_value_overloads"
    record["instances"] = [] if is_count(count) else None
    if is_count(count):
        for index in range(count):
            record["instances"].append({
                "index": index,
                "kind": call(metric, "kind", index),
                "value": call(metric, "value", index),
            })
    return record


def metric_record(metric, name):
    if metric is None:
        return {"name": name, "present": False}
    record = {"name": name, "present": True, **value_record(metric)}
    for field in ("name", "description", "rollup_operation", "metric_type",
                  "metric_subtype", "has_correlation_ids"):
        record["native_" + field] = call(metric, field)
    record["correlation"] = None
    if record["native_has_correlation_ids"] is True:
        try:
            correlation = metric.correlation_ids()
            record["correlation"] = (None if correlation is None
                                     else value_record(correlation))
        except Exception as error:
            record["correlation"] = error_record(error)
    return record


def pc_records(action, metrics):
    """Aggregate per-PC ordinary stall samples from already-detached metrics.

    Keeps the not-issued (overlapping) subset separate from the totals, excludes
    unreadable/mismatched instances instead of zero-filling, and performs a
    source + SASS lookup for every valid PC (no top-k truncation of the data).
    """
    ordinary = sorted(name for name in metrics
                      if name.startswith("smsp__pcsamp_warps_issue_stalled_")
                      and "not_issued" not in name)
    overlapping = sorted(name for name in metrics
                         if name.startswith("smsp__pcsamp_warps_issue_stalled_")
                         and "not_issued" in name)
    samples = defaultdict(lambda: defaultdict(int))
    excluded = []
    for name in ordinary:
        record = metrics[name]
        correlation = record.get("correlation")
        values = record.get("instances")
        ids = correlation.get("instances") if isinstance(correlation, dict) else None
        if not isinstance(values, list) or not isinstance(ids, list):
            excluded.append({"metric": name, "reason": "missing/unreadable instances or correlation"})
            continue
        if len(values) != len(ids):
            excluded.append({"metric": name, "reason": "correlation length mismatch",
                             "value_count": len(values), "id_count": len(ids)})
            continue
        for value, identity in zip(values, ids):
            pc, count = identity["value"], value["value"]
            if not is_count(pc) or not is_number(count) or count < 0:
                excluded.append({"metric": name, "index": value["index"],
                                 "reason": "non-numeric, missing or invalid PC/count"})
                continue
            samples[pc][name] += count

    rows = []
    for pc in sorted(samples):
        row = {"pc": pc, "pc_hex": hex(pc), "ordinary_states": dict(samples[pc]),
               "ordinary_state_samples": sum(samples[pc].values()),
               "source": None, "source_mapped": False}
        try:
            source = action.source_info(pc)
            if source is not None:
                row["source"] = {"file": call(source, "file_name"),
                                 "line": call(source, "line")}
                row["source_mapped"] = (
                    isinstance(row["source"]["file"], str)
                    and bool(row["source"]["file"])
                    and row["source"]["file"] != "?"
                    and is_count(row["source"]["line"])
                    and row["source"]["line"] > 0)
        except Exception as error:
            row["source"] = error_record(error)
        row["sass"] = call(action, "sass_by_pc", pc)
        rows.append(row)
    total = sum(row["ordinary_state_samples"] for row in rows)
    mapped = sum(row["ordinary_state_samples"] for row in rows if row["source_mapped"])
    return {
        "ordinary_state_names": ordinary,
        "overlapping_not_issued_names_excluded_from_totals": overlapping,
        "aggregation_basis": ("All readable finite non-negative ordinary instance "
                              "values with readable integer PC IDs. Raw-value "
                              "aggregation, not a native instance-availability claim."),
        "note": ("Ordinary state samples include selected/not_selected; they are "
                 "not all stalls or durations. All raw metrics remain archived. "
                 "Every valid ordinary PC receives source and SASS lookup; no "
                 "top-k truncation."),
        "excluded_instances": excluded,
        "all_ordinary_instances_usable": bool(ordinary) and not excluded,
        "pc_count": len(rows), "lookup_attempt_count": len(rows),
        "mapped_pc_count": sum(row["source_mapped"] for row in rows),
        "ordinary_state_samples": total, "mapped_ordinary_state_samples": mapped,
        "sample_coverage": mapped / total if total > 0 else None,
        "rows": rows,
    }


PM_PASS_GROUPS_METRIC = "profiler__pmsampler_pass_groups"
PM_PREFIXES = ("pmsampling:", "warpsampling:")


def pm_pass_groups(metrics):
    """Discover PM pass-group membership from the report's own metadata.

    The report records its PM sampling pass groups in the
    ``profiler__pmsampler_pass_groups`` metric: its scalar is the group count and
    each per-group instance value is a comma-separated list of the member metric
    names in that group. This is the authoritative membership — it includes
    Triage members that do NOT carry the ``pmsampling:``/``warpsampling:`` name
    prefix, which a prefix guess silently drops.

    Returns a structured record preserving per-group identity, the de-duplicated
    member set, and the historical prefix-only convenience list (kept as a
    regression). If the metadata metric is absent or unreadable, the limitation
    is recorded and membership falls back to the name prefix — never invented.
    """
    prefix_names = sorted(n for n in metrics if n.startswith(PM_PREFIXES))
    rec = metrics.get(PM_PASS_GROUPS_METRIC)
    if not isinstance(rec, dict) or not rec.get("present"):
        return {
            "source": "name-prefix-fallback",
            "metadata_available": False,
            "group_count": None, "groups": [],
            "members": prefix_names, "member_count": len(prefix_names),
            "convenience_prefix_series": prefix_names,
            "triage_members_recovered": [], "members_absent_from_metrics": [],
            "note": (f"{PM_PASS_GROUPS_METRIC} absent/unreadable; PM membership "
                     "falls back to the pmsampling:/warpsampling: name prefix and "
                     "may undercount pass-group members. No series was invented."),
        }
    groups, members, parse_errors = [], [], []
    instances = rec.get("instances")
    if isinstance(instances, list):
        for inst in instances:
            gi = inst.get("index") if isinstance(inst, dict) else None
            val = inst.get("value") if isinstance(inst, dict) else None
            if isinstance(val, str):
                gmembers = [m for m in (s.strip() for s in val.split(",")) if m]
            else:
                gmembers = []
                parse_errors.append({"group_index": gi,
                                     "reason": "non-string group member list"})
            groups.append({"group_index": gi, "members": gmembers})
            members.extend(gmembers)
    uniq = sorted(set(members))
    return {
        "source": PM_PASS_GROUPS_METRIC,
        "metadata_available": True,
        "group_count": rec.get("value"),
        "groups": groups,
        "members": uniq, "member_count": len(uniq),
        "convenience_prefix_series": prefix_names,
        "triage_members_recovered": [m for m in uniq if not m.startswith(PM_PREFIXES)],
        "members_absent_from_metrics": [m for m in uniq if m not in metrics],
        "parse_errors": parse_errors,
        "note": ("PM series discovered from report pass-group metadata; each "
                 "member's raw record (value/has_value/unit/instances) is "
                 "preserved verbatim. A has_value=false member is unavailable, "
                 "not a measured zero. Device-wide PM samples are not kernel-only "
                 "attribution."),
    }


def capture_action(action, range_index, action_index):
    names = sorted(action.metric_names())
    metrics = {}
    for name in names:
        try:
            metrics[name] = metric_record(action.metric_by_name(name), name)
        except Exception as error:
            metrics[name] = {"name": name, "present_in_metric_names": True,
                             **error_record(error)}
    pm_groups = pm_pass_groups(metrics)
    return {
        "range_index": range_index, "action_index": action_index,
        "name": call(action, "name", action.NameBase_DEMANGLED),
        "function_name": call(action, "name", action.NameBase_FUNCTION),
        "metric_count": len(names), "metric_names": names, "metrics": metrics,
        "availability_checks": {name: name in metrics for name in
                                ("dram__bytes_read.sum", "dram__bytes_write.sum",
                                 "dram__throughput.avg.pct_of_peak_sustained_elapsed")},
        "pc_analysis": pc_records(action, metrics),
        # PM series keyed from the report's own pass-group membership (not a name
        # prefix guess); each member's raw record is preserved. pm_pass_groups
        # records per-group identity and the historical prefix convenience set.
        "pm_series": {name: metrics[name] for name in pm_groups["members"]
                      if name in metrics},
        "pm_pass_groups": pm_groups,
        "rule_results": call(action, "rule_results_as_dicts"),
    }


def capture(report_path, ncu_report_module=None):
    """Load one report and return a detached schema-3 capture dict.

    ``ncu_report_module`` can be injected (CPU tests supply a fake); otherwise
    the native module is imported here — the single, lazy import site.
    """
    if ncu_report_module is None:
        try:
            import ncu_report as ncu_report_module  # the only native import site
        except Exception as error:
            # Fail loudly: never silently fall back to a fake/stub reader in the
            # production path. Point NCU_REPORT_PYTHONPATH at the selected reader
            # (aarch64 ncu-report==2026.3.1 on Orin; NCU 2026.1.1 extras/python on
            # Thor). CPU tests inject the fake module explicitly instead.
            raise RuntimeError(
                f"native ncu_report reader is not importable "
                f"({type(error).__name__}: {error}); set NCU_REPORT_PYTHONPATH to "
                f"the selected reader's module directory — there is no fake/fallback "
                f"reader in the production path"
            ) from error

    report = ncu_report_module.load_report(str(report_path))
    module_file = getattr(ncu_report_module, "__file__", None)
    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": ("Independent detached native read. One report in this fresh "
                  "process; no GPU execution, unit normalization, timing "
                  "inference, or source-producer attribution."),
        "report": {"path": str(report_path)},
        "reader": {"path": str(Path(__file__).resolve()),
                   "native_module": str(module_file) if module_file else None,
                   "python": sys.version, "pid": os.getpid()},
        "num_ranges": report.num_ranges(), "ranges": [], "actions": [],
        "structural_errors": [],
        "interpretation_limits": [
            "Missing metrics are absent from metric_names; missing values, native "
            "exceptions and non-finite values stay distinct from numeric zero.",
            "Native has_value() describes the scalar; its one-argument overload "
            "takes ValueKind. Instance availability is not exposed by this API; "
            "instance kind/value are archived without calling has_value(index).",
            "PM/warp values, native units and correlation units are preserved "
            "verbatim. Indices are not timestamps; replay passes are not aligned.",
            "Dropped-data metric unit disagreements require separate "
            "investigation; no expected label is imposed.",
            "NCU replay durations and anomalous ratios remain native "
            "observations, not independent event latency or clamped limits.",
        ],
    }
    try:
        report_file = Path(report_path)
        if report_file.is_file():
            result["report"]["bytes"] = report_file.stat().st_size
            result["report"]["sha256"] = digest(report_file)
    except Exception as error:
        result["report"]["digest_error"] = error_record(error)

    for range_index in range(result["num_ranges"]):
        try:
            native_range = report.range_by_idx(range_index)
            action_count = native_range.num_actions()
        except Exception as error:
            result["structural_errors"].append({"range_index": range_index,
                                                 **error_record(error)})
            continue
        result["ranges"].append({"range_index": range_index, "num_actions": action_count})
        for action_index in range(action_count):
            try:
                action = native_range.action_by_idx(action_index)
                result["actions"].append(capture_action(action, range_index, action_index))
            except Exception as error:
                result["structural_errors"].append({"range_index": range_index,
                                                     "action_index": action_index,
                                                     **error_record(error)})
    result["num_actions"] = len(result["actions"])
    # A partial capture preserves every successfully-read action alongside the
    # structural_errors in the detached JSON. The CLI (and the JSON file it
    # writes) is the intended consumer of this partial output, for direct
    # diagnostic/recovery inspection; the Python load_capture() consumer is
    # strict fail-loud on the non-zero exit below (see ncu_utils.load_capture).
    result["status"] = ("partial_native_report" if result["structural_errors"]
                        else "captured_native_report")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detached NCU capture (schema 3).")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Must not already exist; earlier captures are preserved.")
    args = parser.parse_args(argv)
    report, output = args.report.resolve(), args.output.resolve()
    if not report.is_file():
        parser.error("--report must be an existing file")
    if output.exists():
        parser.error("--output must not exist; preserve earlier captures")
    # Allow an explicit SDK path (a PYTHONPATH-style os.pathsep list) when
    # isolated Python ignores PYTHONPATH; only absolute existing dirs are added.
    insert_sdk_path(os.environ.get("NCU_REPORT_PYTHONPATH"))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = capture(report)
        exit_code = int(bool(result["structural_errors"]))
    except Exception as error:
        result = {"status": "capture_failed", "report": str(report), **error_record(error)}
        exit_code = 1
    with output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "num_actions": result.get("num_actions")}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
