#!/usr/bin/env python3
"""CPU-only tests for the detached NCU capture seam (schema 3).

Run: python3 -I -m unittest discover -s skills/ncu-report-skill/tests -p 'test_*.py'
All tests inject a fake ``ncu_report`` — no NCU SDK, no GPU.
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parents[0].parent / "ncu-report-skill" / "scripts"
# tests/ is on sys.path via unittest discover; add scripts/ for the modules.
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import ncu_capture  # noqa: E402
import ncu_utils as u  # noqa: E402
import fake_ncu as fk  # noqa: E402


def capture_of(ranges):
    return fk.capture_of(ranges)


class CaptureMechanicsTests(unittest.TestCase):

    def test_enumerate_every_action_not_just_first(self):
        # One range, two actions; a second range with one action -> 3 total.
        ranges = [
            [{"name": "k0", "metrics": {"a": fk.scalar_metric(1)}},
             {"name": "k1", "metrics": {"a": fk.scalar_metric(2)}}],
            [{"name": "k2", "metrics": {"a": fk.scalar_metric(3)}}],
        ]
        cap, _ = capture_of(ranges)
        self.assertEqual(cap["num_actions"], 3)
        names = [u.action_name(a) for a in u.select_actions(cap)]
        self.assertEqual(names, ["k0", "k1", "k2"],
                         "capture must enumerate every range/action, not only action 0")

    def test_five_value_states_distinguishable_and_json_safe(self):
        metrics = {
            "m_nan": fk.scalar_metric(float("nan")),
            "m_inf": fk.scalar_metric(float("inf")),
            "m_zero": fk.scalar_metric(0),
            "m_err": fk.scalar_metric(fk.Boom("boom")),
        }  # "m_absent" intentionally omitted
        cap, _ = capture_of([[{"name": "k", "metrics": metrics}]])
        # Must not crash on non-finite — the sentinels are plain JSON.
        json.dumps(cap, allow_nan=False)
        action = cap["actions"][0]
        self.assertEqual(u.classify_value(u.metric_value(action, "m_nan")), "nan")
        self.assertEqual(u.classify_value(u.metric_value(action, "m_inf")), "+inf")
        self.assertEqual(u.classify_value(u.metric_value(action, "m_zero")), "true-zero")
        self.assertEqual(u.classify_value(u.metric_value(action, "m_err")), "read-error")
        self.assertFalse(u.metric_present(action, "m_absent"))
        self.assertEqual(u.classify_value(u.metric_value(action, "m_absent")), "absent")

    def test_has_value_index_overload_never_called(self):
        metrics = {"a": fk.scalar_metric(1, num_instances=3,
                                         instances=[{"kind": "UINT64", "value": i} for i in range(3)])}
        _, module = capture_of([[{"name": "k", "metrics": metrics}]])
        self.assertEqual(module.has_value_arg_calls, [],
                         "has_value(index)/has_value(ValueKind) must never be called by the capture")

    def test_has_value_kept_separate_from_value(self):
        metrics = {"a": fk.scalar_metric(0, has_value=False)}
        cap, _ = capture_of([[{"name": "k", "metrics": metrics}]])
        action = cap["actions"][0]
        # value is a true zero, but scalar has_value() reports False — distinct axes.
        self.assertEqual(u.classify_value(u.metric_value(action, "a")), "true-zero")
        self.assertIs(u.metric_has_value(action, "a"), False)

    def test_pc_true_zero_sample_not_dropped(self):
        m = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait",
                         [(0x10, 5), (0x20, 0), (0x30, 3)])
        cap, _ = capture_of([[{"name": "k", "metrics": {m["native_name"]: m}}]])
        pcs = {row["pc"] for row in cap["actions"][0]["pc_analysis"]["rows"]}
        self.assertIn(0x20, pcs, "a true-zero PC sample must be preserved, not dropped")

    def test_correlation_length_mismatch_excluded_not_zero_filled(self):
        bad = {
            "native_name": "smsp__pcsamp_warps_issue_stalled_wait",
            "unit": "sample", "num_instances": 3,
            "instances": [{"kind": "UINT64", "value": v} for v in (1, 2, 3)],
            "correlation": {"unit": "sample", "num_instances": 2,
                            "instances": [{"kind": "UINT64", "value": pc} for pc in (0x10, 0x20)]},
        }
        cap, _ = capture_of([[{"name": "k", "metrics": {bad["native_name"]: bad}}]])
        pca = cap["actions"][0]["pc_analysis"]
        reasons = [e["reason"] for e in pca["excluded_instances"]]
        self.assertTrue(any("correlation length mismatch" in r for r in reasons))
        self.assertEqual(pca["pc_count"], 0, "mismatched data must be excluded, not zero-filled")

    def test_not_issued_subset_excluded_from_totals(self):
        ordinary = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait", [(0x10, 4)])
        not_issued = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait_not_issued", [(0x10, 99)])
        cap, _ = capture_of([[{"name": "k", "metrics": {
            ordinary["native_name"]: ordinary, not_issued["native_name"]: not_issued}}]])
        pca = cap["actions"][0]["pc_analysis"]
        self.assertIn("smsp__pcsamp_warps_issue_stalled_wait_not_issued",
                      pca["overlapping_not_issued_names_excluded_from_totals"])
        self.assertEqual(pca["ordinary_state_samples"], 4,
                         "not-issued subset must not be double-counted into the totals")

    def test_no_lineinfo_keeps_pcs_with_source_coverage_zero(self):
        m = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait", [(0x10, 5), (0x20, 7)])
        # No `sources` map -> source_info returns None for every PC.
        cap, _ = capture_of([[{"name": "k", "metrics": {m["native_name"]: m}}]])
        pca = cap["actions"][0]["pc_analysis"]
        self.assertEqual(pca["pc_count"], 2)
        self.assertEqual(pca["mapped_pc_count"], 0)
        self.assertEqual(pca["sample_coverage"], 0.0,
                         "no-lineinfo report keeps PCs; source coverage is 0, not fabricated")

    def test_source_mapping_when_lineinfo_present(self):
        m = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait", [(0x10, 5)])
        action = {"name": "k", "metrics": {m["native_name"]: m},
                  "sources": {0x10: ("kernel.cu", 42)}}
        cap, _ = capture_of([[action]])
        pca = cap["actions"][0]["pc_analysis"]
        self.assertEqual(pca["mapped_pc_count"], 1)
        self.assertEqual(pca["sample_coverage"], 1.0)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.cap, _ = capture_of([
            [{"name": "k0", "metrics": {"a": fk.scalar_metric(1)}},
             {"name": "k1", "metrics": {"a": fk.scalar_metric(2)}}]])

    def test_default_enumerates_all(self):
        self.assertEqual(len(u.select_actions(self.cap)), 2)

    def test_explicit_select(self):
        sel = u.select_actions(self.cap, "r0a1")
        self.assertEqual(len(sel), 1)
        self.assertEqual(u.action_name(sel[0]), "k1")

    def test_flat_index_select(self):
        self.assertEqual(u.action_name(u.select_actions(self.cap, 0)[0]), "k0")

    def test_nonexistent_selector_fails_loud(self):
        with self.assertRaises(u.ActionSelectorError):
            u.select_actions(self.cap, "r9a9")

    def test_unrecognized_selector_fails_loud(self):
        with self.assertRaises(u.ActionSelectorError):
            u.select_actions(self.cap, "garbage")


class PreserveAnomalyTests(unittest.TestCase):
    def test_sample_vs_warp_unit_discrepancy_preserved(self):
        # A dropped-data metric reports unit 'sample'; a direct scalar lookup of
        # a related metric reports 'warp'. The tool must preserve BOTH verbatim.
        dropped = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait",
                               [(0x10, 3)], unit="sample", correlation_unit="sample")
        direct = fk.scalar_metric(2.0, unit="warp")
        cap, _ = capture_of([[{"name": "k", "metrics": {
            dropped["native_name"]: dropped, "smsp__warp_direct": direct}}]])
        action = cap["actions"][0]
        self.assertEqual(u.metric_unit(action, "smsp__pcsamp_warps_issue_stalled_wait"), "sample")
        self.assertEqual(u.metric_unit(action, "smsp__warp_direct"), "warp")

    def test_occupancy_inconsistency_preserved(self):
        # Achieved occupancy exceeding theoretical is a reported anomaly; the
        # capture must keep both values verbatim, not clamp them.
        metrics = {
            "sm__warps_active.avg.pct_of_peak_sustained_active": fk.scalar_metric(71.0),
            "launch__occupancy_limit_warps": fk.scalar_metric(48.0),
        }
        cap, _ = capture_of([[{"name": "k", "metrics": metrics}]])
        action = cap["actions"][0]
        self.assertEqual(u.metric_value(action, "sm__warps_active.avg.pct_of_peak_sustained_active"), 71.0)
        self.assertEqual(u.metric_value(action, "launch__occupancy_limit_warps"), 48.0)


if __name__ == "__main__":
    unittest.main()
