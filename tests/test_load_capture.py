#!/usr/bin/env python3
"""Integration tests for the load_capture subprocess seam (CPU-only).

These run the real capture child in a fresh interpreter against a MINIMAL fake
``ncu_report`` SDK placed on a temporary path, exercising the isolated-child
environment bridging and error surfacing without any NCU SDK or GPU.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import ncu_utils as u  # noqa: E402

MINIMAL_SDK = (
    "class _Report:\n"
    "    def num_ranges(self):\n"
    "        return 0\n"
    "def load_report(path):\n"
    "    return _Report()\n"
)


class LoadCaptureEnvTests(unittest.TestCase):
    def _sdk_and_report(self, d):
        sdk = Path(d) / "sdk"
        sdk.mkdir()
        (sdk / "ncu_report.py").write_text(MINIMAL_SDK)
        report = Path(d) / "fake.ncu-rep"
        report.write_text("x")
        return sdk, report

    def test_pythonpath_is_bridged_to_isolated_child(self):
        # The documented `PYTHONPATH=<extras/python>` setup must reach the -I
        # child, which otherwise ignores PYTHONPATH.
        with tempfile.TemporaryDirectory() as d:
            sdk, report = self._sdk_and_report(d)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(sdk)
            env.pop("NCU_REPORT_PYTHONPATH", None)
            cap = u.load_capture(report, env=env)
            self.assertEqual(cap["num_actions"], 0)
            self.assertEqual(cap["status"], "captured_native_report")

    def test_ncu_report_pythonpath_also_works(self):
        with tempfile.TemporaryDirectory() as d:
            sdk, report = self._sdk_and_report(d)
            env = dict(os.environ)
            env["NCU_REPORT_PYTHONPATH"] = str(sdk)
            env.pop("PYTHONPATH", None)
            cap = u.load_capture(report, env=env)
            self.assertEqual(cap["status"], "captured_native_report")

    def test_multidir_pythonpath_with_relative_entry_still_imports(self):
        # BSI-1: a colon-list PYTHONPATH must reach the child split into entries,
        # with empty/relative parts dropped (a relative part would re-enable cwd
        # import under -I). The real SDK dir is one entry among junk; capture must
        # still succeed and not be defeated by the sibling entries.
        with tempfile.TemporaryDirectory() as d:
            sdk, report = self._sdk_and_report(d)
            other = Path(d) / "other"
            other.mkdir()
            env = dict(os.environ)
            # empty entry (cwd), a relative entry (cwd re-enable hazard), a real
            # sibling dir, and the actual SDK dir — all in one os.pathsep list.
            env["PYTHONPATH"] = os.pathsep.join(
                ["", "relative/sub", str(other), str(sdk)])
            env.pop("NCU_REPORT_PYTHONPATH", None)
            cap = u.load_capture(report, env=env)
            self.assertEqual(cap["status"], "captured_native_report")

    def test_sdk_path_entries_splits_and_filters(self):
        # BSI-1 unit level: the helper keeps only absolute existing dirs, splits
        # on os.pathsep, and drops empty/relative entries (never cwd).
        import ncu_capture as cap
        with tempfile.TemporaryDirectory() as d:
            real = Path(d) / "extras"
            real.mkdir()
            raw = os.pathsep.join(
                ["", "  ", "rel/dir", str(real), str(Path(d) / "missing")])
            entries = cap.sdk_path_entries(raw)
            self.assertEqual(entries, [str(real)])
            self.assertEqual(cap.sdk_path_entries(None), [])
            self.assertEqual(cap.sdk_path_entries(""), [])
            self.assertEqual(cap.sdk_path_entries("."), [])  # cwd never admitted

    def test_missing_sdk_raises_with_structured_detail(self):
        # When the child fails, the error text must surface the archived native
        # error, not an empty string after the colon.
        with tempfile.TemporaryDirectory() as d:
            report = Path(d) / "fake.ncu-rep"
            report.write_text("x")
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env.pop("NCU_REPORT_PYTHONPATH", None)
            with self.assertRaises(u.CaptureError) as ctx:
                u.load_capture(report, env=env)
            self.assertIn("ModuleNotFoundError", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
