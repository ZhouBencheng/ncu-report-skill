#!/usr/bin/env python3
"""Structural guards: the first-action-only bug and the CPU-import blocker are gone."""
import importlib
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


class StructureTests(unittest.TestCase):
    def test_first_action_only_path_removed(self):
        for name in ("ncu_utils.py", "analyze_reports.py",
                     "extract_stall_hotspots.py", "plot_timeline.py", "ncu_capture.py"):
            text = (SCRIPTS / name).read_text()
            self.assertNotIn("action_by_idx(0)", text,
                             f"{name} still collapses to the first action")

    def test_only_capture_imports_native_module(self):
        # ncu_capture may reference ncu_report (lazily, inside a function); no
        # other script may import it at all.
        for name in ("ncu_utils.py", "analyze_reports.py",
                     "extract_stall_hotspots.py", "plot_timeline.py"):
            text = (SCRIPTS / name).read_text()
            self.assertNotIn("import ncu_report", text,
                             f"{name} must not import the native module")
        # ncu_capture imports it lazily, never at module top level.
        cap_text = (SCRIPTS / "ncu_capture.py").read_text()
        top_level = [ln for ln in cap_text.splitlines()
                     if ln.strip() == "import ncu_report"]
        self.assertEqual(top_level, [], "ncu_report must be imported lazily, not at top level")

    def test_modules_import_on_cpu_host(self):
        # No native SDK present in this environment; importing must still work.
        for name in ("ncu_utils", "ncu_capture", "analyze_reports",
                     "extract_stall_hotspots", "plot_timeline"):
            importlib.import_module(name)

    def test_stale_ncu_version_hint_updated(self):
        for name in ("ncu_utils.py", "ncu_capture.py"):
            self.assertNotIn("2026.1.0", (SCRIPTS / name).read_text(),
                             f"{name} still carries the stale 2026.1.0 NCU hint")


if __name__ == "__main__":
    unittest.main()
