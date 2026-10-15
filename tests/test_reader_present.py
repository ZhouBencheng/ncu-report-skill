#!/usr/bin/env python3
"""Reader-present replay + the lineinfo/no-lineinfo PC-mapping distinction.

The lineinfo-distinction check is a CPU test over the frozen comparison artifact
(always runs when present). The real-report replay needs the native aarch64
``ncu-report==2026.3.1`` reader and the ``.ncu-rep`` reports; it is SKIPPED when
the reader wheel or the reports are absent (root-owned / gate B).
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))

import ncu_utils as u  # noqa: E402


def _find_up(relpath):
    for base in HERE.parents:
        cand = base / relpath
        if cand.exists():
            return cand
    return None


LINEINFO_JSON = _find_up("inputs/lineinfo-comparison.json")
REPORT_LINEINFO = _find_up("evidence/device-probe-v1/copy-reduce-lineinfo.ncu-rep")
READER_PYTHON = _find_up("tools/ncu-reader/bin/python")


def _reader_importable():
    if READER_PYTHON is None:
        return False
    try:
        r = subprocess.run([str(READER_PYTHON), "-I", "-c", "import ncu_report"],
                           capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


READER_PRESENT = _reader_importable()


@unittest.skipIf(LINEINFO_JSON is None, "lineinfo-comparison.json not present")
class LineinfoDistinctionTests(unittest.TestCase):
    """The mapped/unmapped PC-row distinction is preserved, not collapsed."""

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(LINEINFO_JSON.read_text())
        cls.by_dir = {r["directory"]: r for r in cls.data["reports"]}

    def test_lineinfo_report_maps_every_pc_row(self):
        acts = self.by_dir["device-probe-v1"]["actions"]
        self.assertEqual([a["pc_rows"] for a in acts], [85, 88])
        for a in acts:
            self.assertEqual(a["source_mapped_rows"], a["pc_rows"],
                             "lineinfo report maps every PC row")

    def test_no_lineinfo_keeps_pc_rows_with_zero_mapping(self):
        acts = self.by_dir["device-probe-no-lineinfo-v1"]["actions"]
        self.assertEqual([a["pc_rows"] for a in acts], [83, 90])
        for a in acts:
            self.assertEqual(a["source_mapped_rows"], 0,
                             "no-lineinfo report keeps PC rows but maps none")
            self.assertGreater(a["pc_rows"], 0,
                               "unmapped PC evidence must not be dropped")


@unittest.skipUnless(READER_PRESENT and REPORT_LINEINFO is not None,
                     "native aarch64 ncu-report reader or report absent (root-owned)")
class RealReportReplayTests(unittest.TestCase):
    """Reader-present: the real two-action report replays deterministically."""

    def test_repeat_fresh_import_matches(self):
        cap1 = u.load_capture(REPORT_LINEINFO, python=str(READER_PYTHON))
        cap2 = u.load_capture(REPORT_LINEINFO, python=str(READER_PYTHON))
        self.assertEqual(cap1["num_ranges"], cap2["num_ranges"])
        self.assertEqual(cap1["num_actions"], cap2["num_actions"])
        names1 = [a.get("name") for a in cap1["actions"]]
        names2 = [a.get("name") for a in cap2["actions"]]
        self.assertEqual(names1, names2)

    def test_real_report_has_seven_groups_31_members(self):
        cap = u.load_capture(REPORT_LINEINFO, python=str(READER_PYTHON))
        for action in cap["actions"]:
            pg = action["pm_pass_groups"]
            self.assertEqual(pg["group_count"], 7)
            self.assertEqual(pg["member_count"], 31)
            self.assertEqual(len(action["pm_series"]), 31)


if __name__ == "__main__":
    unittest.main()
