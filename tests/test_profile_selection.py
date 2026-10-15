#!/usr/bin/env python3
"""CPU-only tests for --profile selection on the report analyzer.

All synthetic identity fixtures below are CPU ROUTING COVERAGE, not hardware
validation: they craft typed identity fields to exercise the selection logic. No
NCU SDK, GPU, or real report is touched — load_capture is monkeypatched to return
a prebuilt detached capture so main() runs fully on a CPU host.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import ncu_utils as u  # noqa: E402
import analyze_reports  # noqa: E402
import fake_ncu as fk  # noqa: E402


def identity_metrics(display=None, cc=None, extra=None):
    m = {}
    if display is not None:
        m[u.IDENTITY_DISPLAY_NAME] = fk.scalar_metric(display)
    if cc is not None:
        m[u.IDENTITY_CC_MAJOR] = fk.scalar_metric(cc[0])
        m[u.IDENTITY_CC_MINOR] = fk.scalar_metric(cc[1])
    m.update(extra or {})
    return m


def capture_with(actions_identities):
    """Build a detached capture; actions_identities is a list of metric dicts."""
    ranges = [[{"name": f"k{i}", "metrics": mm} for i, mm in enumerate(actions_identities)]]
    return fk.capture_of(ranges)[0]


class SelectProfileUnitTests(unittest.TestCase):
    """The pure selection function: auto / confirmed / unverified / conflict."""

    def sel(self, requested, display=None, cc=None):
        cap = capture_with([identity_metrics(display, cc)])
        ident = u.read_identity(u.select_actions(cap)[0])
        return u.select_profile(requested, ident)

    def test_auto_on_qualified_orin_selects_orin(self):
        chosen, status, _ = self.sel("auto", "Orin", (8, 7))
        self.assertEqual((chosen, status), ("orin", "auto-confirmed"))

    def test_auto_unknown_selects_generic_never_b200(self):
        chosen, status, _ = self.sel("auto")  # no identity
        self.assertEqual(chosen, "generic")
        self.assertEqual(status, "auto-generic")
        self.assertNotEqual(chosen, "b200")

    def test_auto_wrong_cc_is_generic_not_orin(self):
        chosen, _, _ = self.sel("auto", "Orin", (8, 6))  # not (8,7)
        self.assertEqual(chosen, "generic")

    def test_explicit_orin_confirmed_on_match(self):
        chosen, status, _ = self.sel("orin", "Orin", (8, 7))
        self.assertEqual((chosen, status), ("orin", "confirmed"))

    def test_explicit_b200_conflicts_with_orin_identity(self):
        chosen, status, _ = self.sel("b200", "Orin", (8, 7))
        self.assertEqual((chosen, status), ("generic", "conflict"))

    def test_explicit_orin_conflicts_with_established_non_orin(self):
        chosen, status, _ = self.sel("orin", "B200", (10, 0))
        self.assertEqual((chosen, status), ("generic", "conflict"))

    def test_explicit_orin_unverified_when_identity_unestablished(self):
        chosen, status, _ = self.sel("orin")  # no identity fields
        self.assertEqual((chosen, status), ("orin", "unverified"))

    def test_explicit_b200_unverified_when_unestablished(self):
        chosen, status, _ = self.sel("b200")
        self.assertEqual((chosen, status), ("b200", "unverified"))

    def test_generic_explicit_never_conflicts(self):
        chosen, status, _ = self.sel("generic", "Orin", (8, 7))
        self.assertEqual((chosen, status), ("generic", "confirmed"))

    def test_read_error_identity_is_not_readable(self):
        cap = capture_with([{u.IDENTITY_DISPLAY_NAME: fk.Boom("boom"),
                             u.IDENTITY_CC_MAJOR: fk.scalar_metric(8),
                             u.IDENTITY_CC_MINOR: fk.scalar_metric(7)}])
        ident = u.read_identity(u.select_actions(cap)[0])
        self.assertIsNone(ident["display_name"])  # read-error is not a value
        chosen, status, _ = u.select_profile("auto", ident)
        self.assertEqual(chosen, "generic")

    # --- explicit b200/thor contradictions (correction 3) ------------------
    def test_explicit_b200_conflicts_with_readable_thor(self):
        chosen, status, _ = self.sel("b200", "NVIDIA Thor", (11, 0))
        self.assertEqual((chosen, status), ("generic", "conflict"))

    def test_explicit_thor_conflicts_with_both_b200_spellings(self):
        for name in ("NVIDIA B200", "B200"):
            chosen, status, _ = self.sel("thor", name, (10, 0))
            self.assertEqual((chosen, status), ("generic", "conflict"), name)

    def test_recognized_other_name_alone_conflicts_even_without_cc(self):
        # A single readable recognized-other-target name is sufficient.
        chosen, status, _ = self.sel("b200", "Orin")  # cc absent
        self.assertEqual((chosen, status), ("generic", "conflict"))

    def test_incompatible_cc_component_alone_conflicts_even_without_name(self):
        chosen, status, _ = self.sel("b200", None, (8, 0))  # name absent, major 8 != 10
        self.assertEqual((chosen, status), ("generic", "conflict"))

    def test_explicit_b200_matched_names_stay_unverified(self):
        for name in ("NVIDIA B200", "B200"):
            chosen, status, _ = self.sel("b200", name, (10, 0))
            self.assertEqual((chosen, status), ("b200", "unverified"), name)

    def test_explicit_thor_matched_stays_unverified(self):
        chosen, status, _ = self.sel("thor", "NVIDIA Thor", (11, 0))
        self.assertEqual((chosen, status), ("thor", "unverified"))

    def test_unfamiliar_name_with_noncontradictory_cc_stays_unverified(self):
        self.assertEqual(self.sel("b200", "Mystery")[:2], ("b200", "unverified"))
        self.assertEqual(self.sel("b200", "Mystery", (10, 0))[:2], ("b200", "unverified"))

    def test_unfamiliar_name_with_incompatible_cc_conflicts(self):
        chosen, status, _ = self.sel("b200", "Mystery", (8, 0))
        self.assertEqual((chosen, status), ("generic", "conflict"))

    def test_auto_on_thor_or_b200_stays_generic(self):
        self.assertEqual(self.sel("auto", "NVIDIA Thor", (11, 0))[:2], ("generic", "auto-generic"))
        self.assertEqual(self.sel("auto", "NVIDIA B200", (10, 0))[:2], ("generic", "auto-generic"))


class IdentityEdgeCaseTests(unittest.TestCase):
    """Availability, non-integral CC, and partial-conflict identity edge cases."""

    def ident(self, metrics):
        cap = capture_with([metrics])
        return u.read_identity(u.select_actions(cap)[0])

    def test_unavailable_display_is_not_readable(self):
        # has_value=False behind a stale "Orin" must NOT establish identity.
        ident = self.ident({
            u.IDENTITY_DISPLAY_NAME: fk.scalar_metric("Orin", has_value=False),
            u.IDENTITY_CC_MAJOR: fk.scalar_metric(8),
            u.IDENTITY_CC_MINOR: fk.scalar_metric(7)})
        self.assertIsNone(ident["display_name"])
        self.assertEqual(u.select_profile("auto", ident)[0], "generic")
        self.assertEqual(u.select_profile("orin", ident)[1], "unverified")

    def test_unavailable_cc_component_is_not_readable(self):
        ident = self.ident({
            u.IDENTITY_DISPLAY_NAME: fk.scalar_metric("Orin"),
            u.IDENTITY_CC_MAJOR: fk.scalar_metric(8, has_value=False),
            u.IDENTITY_CC_MINOR: fk.scalar_metric(7)})
        self.assertIsNone(ident["cc_major"])
        self.assertIsNone(ident["cc"])
        self.assertEqual(u.select_profile("auto", ident)[0], "generic")
        self.assertEqual(u.select_profile("orin", ident)[1], "unverified")

    def test_fractional_cc_not_truncated_into_match(self):
        for cc in ((8.5, 7), (8, 7.9)):
            ident = self.ident({
                u.IDENTITY_DISPLAY_NAME: fk.scalar_metric("Orin"),
                u.IDENTITY_CC_MAJOR: fk.scalar_metric(cc[0]),
                u.IDENTITY_CC_MINOR: fk.scalar_metric(cc[1])})
            self.assertIsNone(ident["cc"], f"fractional CC {cc} must not form a pair")
            self.assertEqual(u.select_profile("auto", ident)[0], "generic")
            self.assertNotEqual(u.select_profile("orin", ident)[1], "confirmed")

    def test_boolean_cc_component_rejected(self):
        ident = self.ident({
            u.IDENTITY_DISPLAY_NAME: fk.scalar_metric("Orin"),
            u.IDENTITY_CC_MAJOR: fk.scalar_metric(True),
            u.IDENTITY_CC_MINOR: fk.scalar_metric(7)})
        self.assertIsNone(ident["cc_major"])

    def test_partial_conflict_major_contradicts_even_if_minor_absent(self):
        ident = self.ident({
            u.IDENTITY_DISPLAY_NAME: fk.scalar_metric("Orin"),
            u.IDENTITY_CC_MAJOR: fk.scalar_metric(10)})  # minor absent
        chosen, status, _ = u.select_profile("orin", ident)
        self.assertEqual((chosen, status), ("generic", "conflict"))
        self.assertEqual(u.select_profile("auto", ident)[0], "generic")

    def test_partial_conflict_minor_contradicts_even_if_major_absent(self):
        ident = self.ident({
            u.IDENTITY_DISPLAY_NAME: fk.scalar_metric("Orin"),
            u.IDENTITY_CC_MINOR: fk.scalar_metric(0)})  # major absent, minor 0 != 7
        chosen, status, _ = u.select_profile("orin", ident)
        self.assertEqual((chosen, status), ("generic", "conflict"))
        self.assertEqual(u.select_profile("auto", ident)[0], "generic")


class ProfileConstantTests(unittest.TestCase):
    def test_b200_entry_is_the_exported_constant_itself(self):
        # The exported constant is preserved, not copied/renamed.
        self.assertIs(u.PROFILE_METRICS["b200"], u.B200_KEY_METRICS)
        self.assertIs(analyze_reports.B200_KEY_METRICS, u.B200_KEY_METRICS)

    def test_orin_list_invents_no_metric(self):
        # Every orin curated name is a plausible NCU metric name (no sentinels).
        for m in u.ORIN_KEY_METRICS:
            self.assertIsInstance(m, str)
            self.assertNotIn(" ", m)


class MainSelectionTests(unittest.TestCase):
    """main() end-to-end on CPU with load_capture monkeypatched."""

    def setUp(self):
        self._orig = analyze_reports.load_capture

    def tearDown(self):
        analyze_reports.load_capture = self._orig

    def _run(self, profile, identities, tag="run"):
        cap = capture_with(identities)
        analyze_reports.load_capture = lambda rep, **kw: cap
        d = tempfile.mkdtemp()
        rep = Path(d) / "r.ncu-rep"
        rep.write_text("x")
        code = analyze_reports.main(["--run-dir", d, "--report", str(rep),
                                     "--tag", tag, "--profile", profile])
        return Path(d) / "analysis", code

    def test_auto_real_orin_writes_orin_sidecar(self):
        analysis, code = self._run("auto", [identity_metrics("Orin", (8, 7))])
        self.assertEqual(code, 0)
        sel = json.loads((analysis / "profile_selection_run_r0a0.json").read_text())
        self.assertEqual(sel["chosen_profile"], "orin")
        self.assertEqual(sel["status"], "auto-confirmed")

    def test_conflict_exits_nonzero_and_falls_to_generic(self):
        analysis, code = self._run("b200", [identity_metrics("Orin", (8, 7))])
        self.assertEqual(code, 1, "a readable-identity conflict must exit nonzero")
        sel = json.loads((analysis / "profile_selection_run_r0a0.json").read_text())
        self.assertEqual(sel["chosen_profile"], "generic")
        self.assertEqual(sel["status"], "conflict")

    def test_conflict_compare_table_uses_generic_not_refused_profile(self):
        # Two conflicting actions downgrade to generic; the cross-action compare
        # table must agree — it must NOT render the refused b200 profile's
        # B200-only metric, which the generic list does not contain.
        analysis, code = self._run(
            "b200", [identity_metrics("Orin", (8, 7)), identity_metrics("Orin", (8, 7))])
        self.assertEqual(code, 1)
        compare_txt = next(analysis.glob("compare_*.txt")).read_text()
        self.assertNotIn("smsp__pcsamp_sample_count", compare_txt,
                         "compare must not show the refused b200 profile's metrics")
        self.assertIn("sm__throughput.avg.pct_of_peak_sustained_elapsed", compare_txt)

    def test_nonconflicting_explicit_profile_compare_table_uses_that_profile(self):
        # No readable identity -> explicit b200 is 'unverified' (not refused), so
        # the compare table legitimately uses the b200 list including its B200-only
        # metric. This guards against over-correcting P3 into always-generic.
        analysis, code = self._run("b200", [identity_metrics(), identity_metrics()])
        self.assertEqual(code, 0)
        compare_txt = next(analysis.glob("compare_*.txt")).read_text()
        self.assertIn("smsp__pcsamp_sample_count", compare_txt)

    def test_unknown_identity_never_b200(self):
        analysis, code = self._run("auto", [identity_metrics()])
        self.assertEqual(code, 0)
        sel = json.loads((analysis / "profile_selection_run_r0a0.json").read_text())
        self.assertEqual(sel["chosen_profile"], "generic")

    def test_mixed_device_actions_get_per_action_selection(self):
        analysis, code = self._run(
            "auto", [identity_metrics("Orin", (8, 7)), identity_metrics()])
        a0 = json.loads((analysis / "profile_selection_run_r0a0.json").read_text())
        a1 = json.loads((analysis / "profile_selection_run_r0a1.json").read_text())
        self.assertEqual(a0["chosen_profile"], "orin")
        self.assertEqual(a1["chosen_profile"], "generic")

    def test_raw_archive_byte_identical_across_profiles(self):
        # The raw metrics_all_* archive is byte-content-equivalent regardless of --profile.
        ids = [identity_metrics("Orin", (8, 7),
                                {"gpu__time_duration.sum": fk.scalar_metric(5, unit="ns")})]
        archives, keys = {}, {}
        for profile in ("generic", "orin", "b200"):
            analysis, _ = self._run(profile, ids)
            archives[profile] = (analysis / "metrics_all_run_r0a0.json").read_bytes()
            keys[profile] = (analysis / "metrics_key_run_r0a0.json").read_text()
        self.assertEqual(archives["generic"], archives["orin"])
        self.assertEqual(archives["orin"], archives["b200"])
        # The curated key view DOES differ by profile (that is the feature).
        self.assertNotEqual(keys["orin"], keys["b200"])

    # --- explicit b200/thor conflicts through real main() (correction 3) ----
    def test_explicit_b200_on_thor_identity_exits_nonzero(self):
        analysis, code = self._run("b200", [identity_metrics("NVIDIA Thor", (11, 0))])
        self.assertEqual(code, 1, "readable Thor identity must conflict with explicit b200")
        sel = json.loads((analysis / "profile_selection_run_r0a0.json").read_text())
        self.assertEqual((sel["chosen_profile"], sel["status"]), ("generic", "conflict"))

    def test_explicit_thor_on_both_b200_spellings_exits_nonzero(self):
        for name in ("NVIDIA B200", "B200"):
            analysis, code = self._run("thor", [identity_metrics(name, (10, 0))], tag=name.replace(" ", "_"))
            self.assertEqual(code, 1, name)
            sel = json.loads(next(analysis.glob("profile_selection_*_r0a0.json")).read_text())
            self.assertEqual((sel["chosen_profile"], sel["status"]), ("generic", "conflict"), name)

    def test_explicit_b200_matched_identity_unverified_exit0(self):
        analysis, code = self._run("b200", [identity_metrics("NVIDIA B200", (10, 0))])
        self.assertEqual(code, 0)
        sel = json.loads((analysis / "profile_selection_run_r0a0.json").read_text())
        self.assertEqual((sel["chosen_profile"], sel["status"]), ("b200", "unverified"))

    def test_auto_thor_stays_generic_exit0(self):
        analysis, code = self._run("auto", [identity_metrics("NVIDIA Thor", (11, 0))])
        self.assertEqual(code, 0)
        sel = json.loads((analysis / "profile_selection_run_r0a0.json").read_text())
        self.assertEqual((sel["chosen_profile"], sel["status"]), ("generic", "auto-generic"))


class MixedConflictMainTests(unittest.TestCase):
    """A mixed, multi-report run: a Thor/B200 conflict must not be masked by a
    non-conflicting action, and must drive the aggregate exit to exactly 1."""

    def setUp(self):
        self._orig = analyze_reports.load_capture

    def tearDown(self):
        analyze_reports.load_capture = self._orig

    def _run_multi(self, profile, reports):
        # reports: list of (tag, identities). Each becomes its own report/range.
        d = tempfile.mkdtemp()
        caps, argv = {}, ["--run-dir", d]
        for tag, ids in reports:
            rep = Path(d) / f"{tag}.ncu-rep"
            rep.write_text(tag)
            caps[str(rep)] = capture_with(ids)
            argv += ["--report", str(rep), "--tag", tag]
        argv += ["--profile", profile]
        analyze_reports.load_capture = lambda rep, **kw: caps[str(rep)]
        return Path(d) / "analysis", analyze_reports.main(argv)

    def test_mixed_conflict_first_spans_two_reports_exit1(self):
        # --profile b200: report "aconflict" reads Thor 11.0 (conflict -> generic);
        # report "bother" has no readable identity (b200 -> unverified, exit 0 alone).
        # Conflict is placed first; the aggregate must still be exactly 1.
        analysis, code = self._run_multi("b200", [
            ("aconflict", [identity_metrics("NVIDIA Thor", (11, 0))]),
            ("bother", [identity_metrics()]),
        ])
        self.assertEqual(code, 1, "the Thor conflict must not be masked by the other action")

        conflict = json.loads((analysis / "profile_selection_aconflict_r0a0.json").read_text())
        self.assertEqual((conflict["chosen_profile"], conflict["status"]), ("generic", "conflict"))
        other = json.loads((analysis / "profile_selection_bother_r0a0.json").read_text())
        self.assertEqual((other["chosen_profile"], other["status"]), ("b200", "unverified"))

        # All raw action archives survive, regardless of the conflict.
        self.assertTrue((analysis / "metrics_all_aconflict_r0a0.json").is_file())
        self.assertTrue((analysis / "metrics_all_bother_r0a0.json").is_file())

        # The cross-action compare uses the chosen (mixed -> generic) diagnostics,
        # NOT the refused b200 profile's B200-only metric.
        compare_txt = next(analysis.glob("compare_*.txt")).read_text()
        self.assertNotIn("smsp__pcsamp_sample_count", compare_txt)
        self.assertIn("sm__throughput.avg.pct_of_peak_sustained_elapsed", compare_txt)

    def test_orin_action_does_not_mask_thor_conflict(self):
        # Under --profile orin: one action is a qualified Orin (confirmed, exit 0
        # alone) and one is a readable Thor (conflict). The aggregate is 1.
        analysis, code = self._run_multi("orin", [
            ("aconflict", [identity_metrics("NVIDIA Thor", (11, 0))]),
            ("bok", [identity_metrics("Orin", (8, 7))]),
        ])
        self.assertEqual(code, 1)
        ok = json.loads((analysis / "profile_selection_bok_r0a0.json").read_text())
        self.assertEqual((ok["chosen_profile"], ok["status"]), ("orin", "confirmed"))


if __name__ == "__main__":
    unittest.main()
