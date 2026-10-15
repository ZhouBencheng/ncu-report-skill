#!/usr/bin/env python3
"""CPU-only tests for the three CLIs' capture-consumer logic.

The CLIs' native seam (load_capture -> fresh child process) is covered by the
capture tests; here we exercise the pure consumer functions directly on detached
captures so no subprocess or NCU SDK is needed.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import ncu_capture  # noqa: E402
import ncu_utils as u  # noqa: E402
import fake_ncu as fk  # noqa: E402
import analyze_reports  # noqa: E402
import extract_stall_hotspots as stall  # noqa: E402
import plot_timeline as plot  # noqa: E402


def capture_of(ranges):
    return fk.capture_of(ranges)[0]


class AnalyzeConsumerTests(unittest.TestCase):
    def test_archives_every_action_with_typed_values(self):
        cap = capture_of([[
            {"name": "k0", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(100, unit="ns"),
                                       "dram__bytes_read.sum": fk.scalar_metric(0)}},
            {"name": "k1", "metrics": {"gpu__time_duration.sum": fk.scalar_metric(200, unit="ns")}}]])
        with tempfile.TemporaryDirectory() as d:
            analysis = Path(d)
            slugs = [analyze_reports.collect_action(a, "run", analysis)
                     for a in u.select_actions(cap)]
            self.assertEqual(len(slugs), 2, "every action must be archived, not just the first")
            # The archive is lossless: unit survives round-trip.
            archived = json.loads((analysis / f"metrics_all_{slugs[0]}.json").read_text())
            self.assertEqual(archived["gpu__time_duration.sum"]["unit"], "ns")

    def test_display_keeps_zero_distinct_from_absent(self):
        cap = capture_of([[{"name": "k", "metrics": {"dram__bytes_read.sum": fk.scalar_metric(0)}}]])
        action = u.select_actions(cap)[0]
        self.assertEqual(u.metric_display(action, "dram__bytes_read.sum"), 0)
        self.assertEqual(u.metric_display(action, "dram__bytes_write.sum"), "<absent>")

    def test_display_keeps_read_error_distinct_from_absent(self):
        # A listed metric whose read fails is <read-error>, not <absent>.
        cap = capture_of([[{"name": "k", "metrics": {"m": fk.Boom("lookup fail")}}]])
        action = u.select_actions(cap)[0]
        self.assertEqual(u.metric_display(action, "m"), "<read-error>")
        self.assertEqual(u.classify_value(u.metric_value(action, "m")), "read-error")
        self.assertEqual(u.metric_display(action, "gone"), "<absent>")

    def test_display_unavailable_scalar_not_shown_as_zero(self):
        # has_value()==False with value 0.0 must render <unavailable>, not 0.
        cap = capture_of([[{"name": "k", "metrics": {
            "avail": fk.scalar_metric(0, has_value=True),
            "unavail": fk.scalar_metric(0.0, has_value=False)}}]])
        action = u.select_actions(cap)[0]
        self.assertEqual(u.metric_display(action, "avail"), 0)
        self.assertEqual(u.metric_display(action, "unavail"), "<unavailable>")

    def test_display_has_value_read_error_not_shown_as_zero(self):
        # BSI-2: has_value() itself read-errors but value()==0. The archived
        # has_value is an error record, so the scalar 0 must NOT render as a
        # bare 0 (indistinguishable from a true zero) — it is <read-error>.
        cap = capture_of([[{"name": "k", "metrics": {
            "hv_err": fk.scalar_metric(0, has_value=fk.Boom("hv read fail"))}}]])
        action = u.select_actions(cap)[0]
        self.assertEqual(u.metric_display(action, "hv_err"), "<read-error>")

    def test_duplicate_tags_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r1, r2 = Path(d) / "a.ncu-rep", Path(d) / "b.ncu-rep"
            r1.write_text("x"); r2.write_text("y")
            with self.assertRaises(SystemExit):
                analyze_reports.main(["--run-dir", d, "--report", str(r1), "--tag", "same",
                                      "--report", str(r2), "--tag", "same"])


class StallConsumerTests(unittest.TestCase):
    def test_rollup_excludes_not_issued_and_keeps_zero(self):
        ordinary = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait", [(0x10, 4), (0x20, 0)])
        not_issued = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait_not_issued", [(0x10, 50)])
        cap = capture_of([[{"name": "k",
                            "metrics": {ordinary["native_name"]: ordinary,
                                        not_issued["native_name"]: not_issued},
                            "sources": {0x10: ("k.cu", 7), 0x20: ("k.cu", 8)}}]])
        action = u.select_actions(cap)[0]
        per_line, unmapped, pca = stall.rollup(action)
        self.assertEqual(pca["ordinary_state_samples"], 4)
        self.assertEqual(unmapped, [], "all PCs mapped when lineinfo present")
        self.assertIn(("k.cu", 8), per_line, "a true-zero line is preserved, not dropped")

    def test_rollup_reports_unmapped_pcs_without_fabricating_lines(self):
        m = fk.pc_metric("smsp__pcsamp_warps_issue_stalled_wait", [(0x10, 5)])
        cap = capture_of([[{"name": "k", "metrics": {m["native_name"]: m}}]])  # no sources
        action = u.select_actions(cap)[0]
        per_line, unmapped, pca = stall.rollup(action)
        self.assertEqual(per_line, {})
        self.assertEqual(len(unmapped), 1)
        self.assertEqual(pca["sample_coverage"], 0.0)


class PlotConsumerTests(unittest.TestCase):
    def test_numeric_series_marks_gaps_not_zero(self):
        detached = [1.0, {"nonfinite": "nan"}, 0, None, {"native_error": {"type": "X"}}]
        series, gaps = plot.numeric_series(detached)
        self.assertEqual(series, [1.0, None, 0.0, None, None],
                         "true zero stays 0.0; nan/absent/error become gaps")
        self.assertEqual(gaps, {"nan": 1, "absent": 1, "read-error": 1})

    def test_ascii_plot_does_not_drop_remainder_samples(self):
        # Full-interval partitioning keeps the tail value; the old floor-sized
        # buckets covered indices 0-3 only and dropped the final 100 (max=1).
        out = "\n".join(plot.ascii_plot([1, 1, 1, 1, 100], "tail", max_rows=2, max_cols=2))
        self.assertIn("max=34", out)

    def test_ascii_plot_gap_bucket_distinct_from_measured_zero(self):
        def bars(lines):
            return [ln.split("|", 1)[1] for ln in lines if "|" in ln]
        gap = plot.ascii_plot([1, None, None, 1], "gap", max_rows=2, max_cols=4)
        zero = plot.ascii_plot([1, 0, 0, 1], "zero", max_rows=2, max_cols=4)
        self.assertTrue(any("." in b for b in bars(gap)),
                        "a gap-only bucket must render as missing, not zero")
        self.assertFalse(any("." in b for b in bars(zero)),
                         "measured zeros are not missing buckets")


if __name__ == "__main__":
    unittest.main()
