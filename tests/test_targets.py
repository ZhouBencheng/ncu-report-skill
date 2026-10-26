import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "helpers"))
# Deliberate test double: production still requires the real native reader.
sys.modules.setdefault("ncu_report", types.ModuleType("ncu_report"))
from ncu_profiles import (IDENTITY_METRICS, canonical_target, metric_evidence, metric_names,
                          route_identity, select_target)
from preflight import collection_options
import analyze_reports
import ncu_utils
import plot_timeline
from plot_timeline import ascii_plot


class Metric:
    def __init__(self, value=0, unit="", available=True, error=False):
        self.data, self.units, self.available, self.error = value, unit, available, error

    def value(self):
        if self.error:
            raise RuntimeError("unreadable")
        return self.data

    def unit(self):
        return self.units

    def has_value(self):
        return self.available


class Action:
    NameBase_DEMANGLED, NameBase_FUNCTION, NameBase_MANGLED = range(3)

    def __init__(self, metrics=None):
        self.metrics = metrics or {}

    def metric_names(self):
        return list(self.metrics)

    def __getitem__(self, name):
        return self.metrics[name]

    def name(self, base):
        return "test_kernel"


def action_for(name, major, minor):
    return Action({key: Metric(value) for key, value in zip(IDENTITY_METRICS, (name, major, minor))
                   if value is not None})


class TargetTests(unittest.TestCase):
    def test_aliases_and_compiler_target_separation(self):
        for value, expected in (("Jetson AGX Orin", "sm87"), ("AGX-Thor", "sm110"),
                                ("GeForce RTX 4090", "sm89"), ("sm_100", "sm100")):
            self.assertEqual(canonical_target(value), expected)
        for value in ("sm110a", "sm100f", "Blackwell", "futuregpu"):
            with self.assertRaises(ValueError):
                canonical_target(value)

    def test_real_device_identities(self):
        for name, major, minor, target in (("Orin", 8, 7, "sm87"), ("NVIDIA Thor", 11, 0, "sm110"),
                                          ("NVIDIA GeForce RTX 4090", 8, 9, "sm89"),
                                          ("NVIDIA B200", 10, 0, "sm100"), ("B300", 10, 3, "sm103")):
            self.assertEqual(select_target(action_for(name, major, minor))["target"], target)

    def test_unknown_never_defaults_to_b200(self):
        self.assertEqual(select_target(Action())["target"], "generic")
        self.assertEqual(select_target(action_for("future", 15, 0))["target"], "generic")
        self.assertNotIn("smsp__sass_inst_executed_op_utcmma.sum", metric_names("generic"))

    def test_identity_conflicts_and_partial_cc(self):
        with self.assertRaises(ValueError):
            select_target(action_for("Orin", 11, 0))
        with self.assertRaises(ValueError):
            select_target(action_for("Orin", 8, 7), "b200")
        with self.assertRaises(ValueError):
            select_target(action_for(None, 8, None), "b200")
        selection = select_target(Action(), "thor")
        self.assertEqual(selection["status"], "explicit-unverified")
        self.assertIsNone(selection["runtime_compute_capability"])

    def test_unavailable_and_fractional_identity_not_used(self):
        a = action_for(None, 10, 0)
        a.metrics[IDENTITY_METRICS[1]] = Metric(10, available=False)
        self.assertEqual(select_target(a)["target"], "generic")
        self.assertEqual(select_target(action_for(None, 8.9, 9))["target"], "generic")

    def test_b200_metric_roster_preserved_and_other_targets_scoped(self):
        expected = json.loads((ROOT / "tests/b200-metrics-baseline.json").read_text())
        self.assertEqual(metric_names("sm100"), expected)
        self.assertEqual(metric_names("sm103"), expected)
        self.assertEqual(ncu_utils.KEY_METRICS, expected)
        for target in ("sm87", "sm89"):
            self.assertFalse(any("utcmma" in n or "mem_tensor" in n for n in metric_names(target)))
        self.assertIn("sm__mem_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", metric_names("sm110"))

    def test_zero_missing_unavailable_read_error_and_units(self):
        action = Action({"zero": Metric(0, "%"), "missing_scalar": Metric(5, available=False),
                         "error": Metric(error=True), "ratio": Metric(0.3, "ratio")})
        records = metric_evidence(action, ["zero", "absent", "missing_scalar", "error", "ratio"])
        self.assertEqual(records["zero"], {"status": "present", "value": 0, "unit": "%"})
        self.assertEqual(records["absent"]["status"], "not_in_report")
        self.assertEqual(records["missing_scalar"]["status"], "scalar_unavailable")
        self.assertEqual(records["error"]["status"], "read_error")
        self.assertEqual(records["error"]["unit"], "")
        self.assertEqual(records["ratio"]["unit"], "ratio")

    def test_read_error_retains_accessible_unit(self):
        records = metric_evidence(Action({"error": Metric(unit="ns", error=True)}), ["error"])
        self.assertEqual(records["error"]["status"], "read_error")
        self.assertEqual(records["error"]["unit"], "ns")

    def test_older_reader_rules_and_timelines_are_unavailable(self):
        self.assertIsNone(ncu_utils.rule_speedups(Action()))
        self.assertIsNone(ncu_utils.warp_state_timeline(Action(), 4))

        class PartialOldReader(Action):
            def rule_results(self):
                raise NotImplementedError

            def timed_warp_samples(self):
                raise RuntimeError("API unavailable for this report")

            def timelines(self):
                raise NotImplementedError

        old = PartialOldReader()
        self.assertIsNone(ncu_utils.rule_speedups(old))
        self.assertIsNone(ncu_utils.timed_warp_samples(old))
        self.assertIsNone(ncu_utils.warp_state_timeline(old, 4))

    def test_timeline_list_displays_missing_reader_api_as_unavailable(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as stdout:
            root = Path(directory)
            report = root / "old-reader.ncu-rep"
            report.touch()
            argv = ["plot_timeline.py", "--run-dir", str(root), "--report", str(report),
                    "--tag", "old", "--list"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(plot_timeline, "select_actions", return_value=(None, [Action()])):
                plot_timeline.main()
            self.assertIn("unavailable timed warp samples", stdout.getvalue())
            self.assertNotIn("0 timed warp samples", stdout.getvalue())

    def test_missing_timeline_samples_are_not_zero(self):
        self.assertIn("unavailable", " ".join(ascii_plot([None, None], "pm")))
        self.assertIn("all zero", " ".join(ascii_plot([0, 0], "pm")))
        self.assertIn("?", " ".join(ascii_plot([0, None, 2], "pm")))

    def test_memory_only_active_window_preserves_unavailable_samples(self):
        unknown = [("TPC.Memory.x", None, [None, 10, 0])]
        measured = [("TPC.Memory.x", None, [0, 10, 0])]
        self.assertEqual(plot_timeline._active_samples(unknown, 3), [0, 1])
        self.assertEqual(plot_timeline._active_samples(measured, 3), [1])
        with mock.patch.object(plot_timeline, "pm_series",
                               return_value=([0, 1000, 2000], unknown[0][2])):
            rendered = " ".join(plot_timeline.plot_replays(Action(), ["TPC.Memory.x"], 4, 8))
        self.assertIn("samples 0..2 of 3", rendered)
        self.assertIn("?", rendered)

    def test_collection_recipe_is_discovered_and_jetson_clock_is_explicit(self):
        options = collection_options("sm87", "basic full", "SourceCounters", "--import-source --clock-control")
        self.assertIn("full", options)
        self.assertEqual(options[-2:], ["--clock-control", "none"])
        self.assertNotIn("boost", collection_options("sm89", "full", "", "--clock-control"))
        self.assertEqual(collection_options("generic", "basic", "LaunchStats Occupancy", ""),
                         ["--section", "LaunchStats", "--section", "Occupancy"])

    def test_l4t_clock_recipe_does_not_invent_device_identity(self):
        selection = route_identity({"name": None, "cc_major": None, "cc_minor": None})
        self.assertEqual(selection["target"], "generic")
        options = collection_options(selection["target"], "full", "", "--clock-control",
                                     l4t_observed=True)
        self.assertEqual(options[-2:], ["--clock-control", "none"])

    def test_analyzer_preserves_values_and_writes_availability_evidence(self):
        action = action_for("NVIDIA B200", 10, 0)
        action.metrics.update({n: Metric(i, "unit") for i, n in enumerate(metric_names("sm100"))})
        action.metrics["uncurated__raw_metric.sum"] = Metric(17, "raw-unit")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)
            key = analyze_reports.collect(action, "baseline", "b200", out)
            for i, name in enumerate(metric_names("sm100")):
                self.assertEqual(key[name], i)
            evidence = json.loads((out / "metrics_evidence_b200.json").read_text())
            self.assertEqual(evidence["target"]["target"], "sm100")
            self.assertEqual(evidence["metrics"][metric_names("sm100")[0]]["status"], "present")
            archive = json.loads((out / "metrics_all_b200.json").read_text())
            self.assertIn({"name": "uncurated__raw_metric.sum", "value": 17, "unit": "raw-unit"}, archive)
            self.assertIn("Unavailable", (out / "rules_b200.txt").read_text())

    def test_analyzer_key_and_comparison_keep_nonpresent_states_distinct(self):
        names = metric_names("sm100")
        action = action_for("NVIDIA B200", 10, 0)
        action.metrics.update({names[0]: Metric(0, "count"),
                               names[1]: Metric(4, "count", available=False),
                               names[2]: Metric(error=True)})
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)
            key = analyze_reports.collect(action, "states", "states", out)
            self.assertEqual(key[names[0]], 0)
            self.assertEqual(key["__metric_status__"][names[0]], "present")
            self.assertEqual(key["__metric_status__"][names[1]], "scalar_unavailable")
            self.assertEqual(key["__metric_status__"][names[2]], "read_error")
            self.assertEqual(key["__metric_status__"][names[3]], "not_in_report")
            text = (out / "metrics_key_states.txt").read_text()
            self.assertIn("= 0 count", text)
            self.assertIn("<scalar_unavailable>", text)
            self.assertIn("<read_error>", text)

            other = {names[1]: None, "__metric_status__": {names[1]: "not_in_report"},
                     "__metric_units__": {names[1]: None}}
            analyze_reports.compare({"states": key, "other": other}, out)
            comparison = (out / "compare_states_vs_other.txt").read_text()
            self.assertIn("<scalar_unavailable>", comparison)
            self.assertIn("<not_in_report>", comparison)

    def test_comparison_rejects_different_units(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            analyze_reports.compare({"a": {"metric": 2, "__metric_units__": {"metric": "%"}},
                                     "b": {"metric": 2, "__metric_units__": {"metric": "ratio"}}}, out)
            self.assertIn("incompatible units", (out / "compare_a_vs_b.txt").read_text())

    def test_comparison_withholds_values_when_unit_metadata_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            analyze_reports.compare({"known": {"metric": 2, "__metric_units__": {"metric": "ns"}},
                                     "unknown": {"metric": 2000, "__metric_units__": {"metric": None}}}, out)
            text = (out / "compare_known_vs_unknown.txt").read_text()
            self.assertIn("unit metadata unavailable for unknown", text)
            self.assertIn("values not compared", text)
            self.assertNotIn("2000", text)

    def test_real_report_scalar_fixtures(self):
        fixture = json.loads((ROOT / "tests/fixtures/device-scalars.json").read_text())
        self.assertEqual({report["target"] for report in fixture["reports"]}, {"sm87", "sm89", "sm110"})
        for report in fixture["reports"]:
            receipt = json.loads((ROOT / "reference/architectures" / report["target"] / "validation.json").read_text())
            self.assertIn(report["report_sha256"], {entry["sha256"] for entry in receipt["profiling"]})
            self.assertEqual(report["reader_version"], receipt["environment"]["reader"])
            for action in report["actions"]:
                native = Action({n: Metric(r["value"], r.get("unit"), available=r["state"] == "available")
                                 for n, r in action["metrics"].items()})
                self.assertEqual(select_target(native)["target"], report["target"])
                records = metric_evidence(native, metric_names(report["target"]))
                for name, record in records.items():
                    original = action["metrics"].get(name)
                    if original and original["state"] == "available":
                        self.assertEqual(record["value"], original["value"])
                        self.assertEqual(record["unit"], original.get("unit"))
                    elif original is None:
                        self.assertEqual(record["status"], "not_in_report")

    def test_producer_version_is_separate_from_reader(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory)
            analyze_reports.collect(action_for("Orin", 8, 7), "report", "orin", out,
                                    collector_version="2025.2.1")
            data = json.loads((out / "metrics_evidence_orin.json").read_text())
            self.assertEqual(data["collector"]["version"], "2025.2.1")
            self.assertEqual(data["collector"]["source"], "collection-record argument")
            self.assertNotEqual(data["reader"]["version"], "2025.2.1")

    def test_pm_names_keep_old_and_new_report_formats(self):
        action = Action({name: types.SimpleNamespace(num_instances=lambda: 3)
                         for name in ("pmsampling:sm__cycles_active.avg",
                                      "TPC.TriageCompute.sm__cycles_active.avg", "ordinary__counter.avg")})
        self.assertEqual(ncu_utils.pm_series_names(action),
                         ["TPC.TriageCompute.sm__cycles_active.avg", "pmsampling:sm__cycles_active.avg"])

    def test_rule_speedup_order_is_preserved(self):
        def rule(value):
            return types.SimpleNamespace(has_speedup_estimation=lambda: True,
                speedup_estimation=lambda: {"speedup": value, "type": 2}, has_rule_message=lambda: True,
                rule_message=lambda: {"message": "observed", "type": 2}, section_identifier=lambda: "SOL",
                rule_identifier=lambda: "rule")
        action = types.SimpleNamespace(rule_results=lambda: [rule(3), rule(12)])
        self.assertEqual([r["speedup"] for r in ncu_utils.rule_speedups(action)], [12, 3])


if __name__ == "__main__":
    unittest.main()
