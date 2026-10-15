#!/usr/bin/env python3
"""CPU-only tests for PM pass-group discovery and value/unit fidelity.

All fixtures inject a fake ``ncu_report`` — no NCU SDK, no GPU. The pass-group
fixture mirrors the real report's structure (seven groups, 31 members: 16 with
the pmsampling: prefix + 15 Triage members without it).
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import ncu_capture  # noqa: E402
import ncu_utils as u  # noqa: E402
import fake_ncu as fk  # noqa: E402


def capture_of(ranges):
    return fk.capture_of(ranges)[0]


PREFIXED = [f"pmsampling:smsp__warps_issue_stalled_{i:02d}.avg" for i in range(16)]
TRIAGE = [f"Triage{i:02d}.sm__counter_{i:02d}.pct" for i in range(15)]


def pass_group_action(unavailable_triage=False):
    """Build an action whose profiler__pmsampler_pass_groups declares 7 groups /
    31 members (16 prefixed + 15 Triage), each member present in metrics."""
    members = PREFIXED + TRIAGE  # 31
    # Distribute 31 members across 7 groups deterministically.
    groups = [[] for _ in range(7)]
    for idx, name in enumerate(members):
        groups[idx % 7].append(name)
    group_values = [",".join(g) for g in groups]
    pass_metric = {
        "native_name": "profiler__pmsampler_pass_groups", "value": 7,
        "has_value": True, "unit": "", "num_instances": 7,
        "instances": [{"kind": "UINT64", "value": gv} for gv in group_values],
    }
    metrics = {"profiler__pmsampler_pass_groups": pass_metric}
    for name in PREFIXED:
        metrics[name] = fk.scalar_metric(1.0, unit="%", num_instances=2,
                                         instances=[{"kind": "DOUBLE", "value": 1.0},
                                                    {"kind": "DOUBLE", "value": 2.0}])
    for i, name in enumerate(TRIAGE):
        if unavailable_triage and i == 0:
            metrics[name] = fk.scalar_metric(0.0, unit="%", has_value=False)
        else:
            metrics[name] = fk.scalar_metric(float(i), unit="%")
    return {"name": "k", "metrics": metrics}


class PmPassGroupTests(unittest.TestCase):
    def test_discovers_seven_groups_and_31_members(self):
        cap = capture_of([[pass_group_action()]])
        pg = cap["actions"][0]["pm_pass_groups"]
        self.assertTrue(pg["metadata_available"])
        self.assertEqual(pg["source"], "profiler__pmsampler_pass_groups")
        self.assertEqual(pg["group_count"], 7)
        self.assertEqual(len(pg["groups"]), 7)
        self.assertEqual(pg["member_count"], 31)
        self.assertEqual(len(pg["triage_members_recovered"]), 15)

    def test_pm_series_includes_all_31_members_not_just_prefix(self):
        cap = capture_of([[pass_group_action()]])
        action = cap["actions"][0]
        self.assertEqual(len(action["pm_series"]), 31)
        # The 15 Triage members (no pmsampling: prefix) are no longer dropped.
        for name in TRIAGE:
            self.assertIn(name, action["pm_series"])

    def test_sixteen_series_convenience_view_preserved(self):
        cap = capture_of([[pass_group_action()]])
        pg = cap["actions"][0]["pm_pass_groups"]
        self.assertEqual(sorted(pg["convenience_prefix_series"]), sorted(PREFIXED))
        self.assertEqual(len(pg["convenience_prefix_series"]), 16)

    def test_pm_series_members_are_the_raw_records(self):
        # Device-wide PM samples are not promoted to kernel-only attribution:
        # each pm_series entry is the member's own raw record, not rescaled.
        cap = capture_of([[pass_group_action()]])
        action = cap["actions"][0]
        name = PREFIXED[0]
        self.assertEqual(action["pm_series"][name], action["metrics"][name])
        self.assertIn("not kernel-only attribution", action["pm_pass_groups"]["note"])

    def test_has_value_false_triage_is_unavailable_not_zero(self):
        cap = capture_of([[pass_group_action(unavailable_triage=True)]])
        action = cap["actions"][0]
        name = TRIAGE[0]
        self.assertIn(name, action["pm_series"])
        # Distinct: value is a true zero, but has_value False -> <unavailable>.
        self.assertEqual(u.classify_value(u.metric_value(action, name)), "true-zero")
        self.assertEqual(u.metric_display(action, name), "<unavailable>")

    def test_fallback_when_pass_group_metadata_absent(self):
        # No profiler__pmsampler_pass_groups metric: fall back to the name prefix
        # and RECORD the limitation; never invent a series.
        metrics = {name: fk.scalar_metric(1.0, unit="%") for name in PREFIXED}
        cap = capture_of([[{"name": "k", "metrics": metrics}]])
        pg = cap["actions"][0]["pm_pass_groups"]
        self.assertFalse(pg["metadata_available"])
        self.assertEqual(pg["source"], "name-prefix-fallback")
        self.assertEqual(sorted(pg["members"]), sorted(PREFIXED))
        self.assertIn("undercount", pg["note"])


class ValueUnitFidelityTests(unittest.TestCase):
    def test_five_value_states_distinct_in_curated_view(self):
        metrics = {
            "m_val": fk.scalar_metric(3.5),
            "m_zero": fk.scalar_metric(0),
            "m_nan": fk.scalar_metric(float("nan")),
            "m_err": fk.Boom("boom"),
            "m_unavail": fk.scalar_metric(0.0, has_value=False),
        }  # m_absent omitted
        cap = capture_of([[{"name": "k", "metrics": metrics}]])
        a = cap["actions"][0]
        self.assertEqual(u.metric_display(a, "m_val"), 3.5)
        self.assertEqual(u.metric_display(a, "m_zero"), 0)
        self.assertEqual(u.metric_display(a, "m_nan"), "<nan>")
        self.assertEqual(u.metric_display(a, "m_err"), "<read-error>")
        self.assertEqual(u.metric_display(a, "m_unavail"), "<unavailable>")
        self.assertEqual(u.metric_display(a, "m_absent"), "<absent>")

    def test_cross_unit_values_preserved_not_normalized(self):
        # Same metric name, different native units in two actions: the capture
        # preserves BOTH units verbatim and never normalizes/compares them.
        a0 = {"name": "k0", "metrics": {"x": fk.scalar_metric(1000, unit="ns")}}
        a1 = {"name": "k1", "metrics": {"x": fk.scalar_metric(2.0, unit="us")}}
        cap = capture_of([[a0, a1]])
        acts = u.select_actions(cap)
        self.assertEqual(u.metric_unit(acts[0], "x"), "ns")
        self.assertEqual(u.metric_unit(acts[1], "x"), "us")
        # Raw values are untouched (no silent conversion to a common unit).
        self.assertEqual(u.metric_value(acts[0], "x"), 1000)
        self.assertEqual(u.metric_value(acts[1], "x"), 2.0)

    def test_compare_suppresses_mismatched_and_unknown_units(self):
        import tempfile
        import analyze_reports as ar
        # Same metric, four actions: ns / us / ns / unknown(None) — must NOT be
        # shown as directly comparable bare numbers.
        actions = [
            {"name": "k0", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(1000, unit="ns")}},
            {"name": "k1", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(1.0, unit="us")}},
            {"name": "k2", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(0, unit="ns")}},
            {"name": "k3", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(2, unit=None)}},
        ]
        cap = capture_of([actions])
        collected = {f"s{i}": a for i, a in enumerate(u.select_actions(cap))}
        with tempfile.TemporaryDirectory() as d:
            ar.compare(collected, Path(d), ["gpu__time_duration.sum"])
            out = next(Path(d).glob("compare_*.txt")).read_text()
        row = [ln for ln in out.splitlines() if ln.startswith("gpu__time_duration.sum")][0]
        self.assertIn("not comparable", row, "mismatched/unknown units must be flagged")
        self.assertIn("[ns]", row)
        self.assertIn("[us]", row)
        # Units are attached — a bare comparable "1000   1" row must not appear.
        self.assertNotRegex(row, r"\s1000\s+1\s")

    def test_compare_unreadable_unit_does_not_raise(self):
        import tempfile
        import analyze_reports as ar
        # One action's native unit read errors (an unhashable dict); the other is
        # a readable "ns". main/compare must finish without raising, keep the
        # values, and flag the row non-comparable (the error is not a known unit).
        actions = [
            {"name": "k0", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(1000, unit="ns")}},
            {"name": "k1", "metrics": {"gpu__time_duration.sum":
                                       fk.scalar_metric(1, unit=fk.Boom("SYNTHETIC read failure"))}},
        ]
        cap = capture_of([actions])
        acts = u.select_actions(cap)
        # The capture preserved the native unit error in the raw archive.
        self.assertFalse(u.unit_is_known(u.metric_unit(acts[1], "gpu__time_duration.sum")))
        collected = {f"s{i}": a for i, a in enumerate(acts)}
        with tempfile.TemporaryDirectory() as d:
            ar.compare(collected, Path(d), ["gpu__time_duration.sum"])  # must not raise
            out = next(Path(d).glob("compare_*.txt")).read_text()
        row = [ln for ln in out.splitlines() if ln.startswith("gpu__time_duration.sum")][0]
        self.assertIn("not comparable", row)
        self.assertIn("[ns]", row)
        self.assertIn("unit: read-error", row)

    def test_curated_txt_annotates_native_unit_per_value(self):
        import tempfile
        import analyze_reports as ar
        metrics = {
            "m_ns": fk.scalar_metric(1000, unit="ns"),
            "m_zero": fk.scalar_metric(0, unit="ns"),            # true zero + unit
            "m_dimless": fk.scalar_metric(2.0, unit=""),          # readable empty unit
            "m_nounit": fk.scalar_metric(3.0, unit=None),         # missing unit
            "m_unitbad": fk.scalar_metric(4.0, unit=fk.Boom("unit boom")),  # unreadable
            "m_unavail": fk.scalar_metric(0.0, unit="ns", has_value=False),
        }
        cap = capture_of([[{"name": "k", "metrics": metrics}]])
        action = cap["actions"][0]
        order = list(metrics.keys())
        with tempfile.TemporaryDirectory() as d:
            analysis = Path(d)
            ar.collect_action(action, "run", analysis, order, None)
            txt = (analysis / "metrics_key_run_r0a0.txt").read_text()
        def row(name):
            return [ln for ln in txt.splitlines() if ln.startswith(name)][0]
        # Value state token stays intact AND the native unit is annotated.
        self.assertIn("= 1000 [ns]", row("m_ns"))
        self.assertIn("= 0 [ns]", row("m_zero"))              # true zero, not unavailable
        self.assertIn("[unit: empty]", row("m_dimless"))      # empty distinguishable
        self.assertIn("[unit: unknown]", row("m_nounit"))     # missing distinguishable
        self.assertIn("[unit: read-error]", row("m_unitbad"))
        self.assertIn("<unavailable>", row("m_unavail"))      # state token preserved

    def test_curated_json_documents_lossy_projection(self):
        import json as _json
        import tempfile
        import analyze_reports as ar
        metrics = {"m": fk.scalar_metric(1000, unit="ns")}
        cap = capture_of([[{"name": "k", "metrics": metrics}]])
        action = cap["actions"][0]
        with tempfile.TemporaryDirectory() as d:
            analysis = Path(d)
            ar.collect_action(action, "run", analysis, ["m"], None)
            key = _json.loads((analysis / "metrics_key_run_r0a0.json").read_text())
        self.assertIn("lossy scalar convenience projection", key["__projection__"])
        self.assertIn("metrics_all_run_r0a0.json", key["__projection__"])

    def test_compare_shows_shared_unit_rows_without_suppression(self):
        import tempfile
        import analyze_reports as ar
        actions = [
            {"name": "k0", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(1000, unit="ns")}},
            {"name": "k1", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(2000, unit="ns")}},
        ]
        cap = capture_of([actions])
        collected = {f"s{i}": a for i, a in enumerate(u.select_actions(cap))}
        with tempfile.TemporaryDirectory() as d:
            ar.compare(collected, Path(d), ["gpu__time_duration.sum"])
            out = next(Path(d).glob("compare_*.txt")).read_text()
        row = [ln for ln in out.splitlines() if ln.startswith("gpu__time_duration.sum")][0]
        self.assertNotIn("not comparable", row, "a shared known unit is comparable")
        self.assertIn("[ns]", row)


if __name__ == "__main__":
    unittest.main()
