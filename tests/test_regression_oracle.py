#!/usr/bin/env python3
"""Cross-check the consumer logic against the authoritative -v2 oracle captures.

These schema-3 captures were produced independently on real Thor hardware and
live in the parent campaign's ``analysis/`` directory. They are read-only here.
When the parent campaign is not present (e.g. a delivered standalone skill), the
checks skip cleanly — the fake-driven tests remain the portable gate.
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import ncu_utils as u  # noqa: E402

# HERE = <campaign>/workspace/skills/ncu-report-skill/tests
#   parents[0]=ncu-report-skill  parents[1]=skills  parents[2]=workspace
#   parents[3]=<campaign>
CAMPAIGN = HERE.parents[3] if len(HERE.parents) >= 4 else None
ORACLE = (CAMPAIGN / "analysis") if CAMPAIGN else None


def load_oracle(name):
    if not ORACLE or not (ORACLE / name).is_file():
        raise unittest.SkipTest(f"oracle capture {name} not present (parent campaign absent)")
    return json.loads((ORACLE / name).read_text())


class RegressionOracleTests(unittest.TestCase):
    def test_baseline_two_actions_33_pm_series_each(self):
        cap = load_oracle("native-old-baseline-v2.json")
        actions = u.select_actions(cap)
        self.assertEqual(len(actions), 2, "baseline report must keep BOTH actions")
        for a in actions:
            self.assertEqual(len(a.get("pm_series") or {}), 33)

    def test_candidate_one_action_279_pcs_10245_samples(self):
        cap = load_oracle("native-old-candidate-v2.json")
        actions = u.select_actions(cap)
        self.assertEqual(len(actions), 1)
        pca = actions[0]["pc_analysis"]
        self.assertEqual(len(actions[0]["pm_series"]), 33)
        self.assertEqual(pca["pc_count"], 279)
        self.assertEqual(pca["ordinary_state_samples"], 10245)
        self.assertEqual(pca["sample_coverage"], 1.0)

    def test_total_99_pm_series_across_reports(self):
        base = load_oracle("native-old-baseline-v2.json")
        cand = load_oracle("native-old-candidate-v2.json")
        total = sum(len(a["pm_series"]) for a in u.select_actions(base))
        total += sum(len(a["pm_series"]) for a in u.select_actions(cand))
        self.assertEqual(total, 99, "66 baseline + 33 candidate = 99 PM/warp series")

    def test_no_lineinfo_keeps_pcs_coverage_zero(self):
        cap = load_oracle("native-probe-no-lineinfo-v2.json")
        for a in u.select_actions(cap):
            pca = a["pc_analysis"]
            self.assertGreater(pca["pc_count"], 0, "no-lineinfo report still retains PCs")
            self.assertEqual(pca["sample_coverage"], 0.0, "source coverage is 0, not fabricated")


if __name__ == "__main__":
    unittest.main()
