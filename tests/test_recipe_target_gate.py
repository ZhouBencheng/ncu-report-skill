#!/usr/bin/env python3
"""CPU-only tests for the Orin recipe's own `gate_target_json.py` helper.

These exercise the ACTUAL helper heredoc extracted verbatim from
``references/10-orin-collector-reader.md`` — not a reimplementation — against the
preserved real target stdout (identity/check/profile/timing) and against copied
one-field negative mutations. The recipe's bash fences are syntax-checked with
``bash -n``; their collection commands are never executed (no GPU/NCU/SSH/install).
"""
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = HERE.parent / "references" / "10-orin-collector-reader.md"
EVIDENCE = HERE.parents[2] / "evidence" / "device-probe-v1"

# Preserved real target stdout per mode. profile is NCU-wrapped (the JSON record
# is surrounded by NCU's own ==PROF==/==WARNING== lines).
REAL_STDOUT = {
    "identity": EVIDENCE / "identity.stdout",
    "check": EVIDENCE / "correctness.stdout",
    "timing": EVIDENCE / "timing.stdout",
    "profile": EVIDENCE / "ncu-profile.stdout",
}


def extract_helper_source():
    """Return the exact body of the `gate_target_json.py` heredoc in the recipe."""
    text = RECIPE.read_text().splitlines()
    start = None
    for i, line in enumerate(text):
        if line.startswith('cat > "$OUT/gate_target_json.py" <<\'PY\''):
            start = i + 1
            break
    assert start is not None, "gate_target_json.py heredoc not found in recipe"
    body = []
    for line in text[start:]:
        if line == "PY":
            return "\n".join(body) + "\n"
        body.append(line)
    raise AssertionError("unterminated gate_target_json.py heredoc")


def extract_one_json(text):
    """Extract the single top-level JSON object from (possibly wrapped) stdout."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise AssertionError("no top-level JSON object found")


def bash_fences():
    """Yield each ```bash fenced code block in the recipe."""
    text = RECIPE.read_text().splitlines()
    block, inside = [], False
    for line in text:
        if not inside and line.strip() == "```bash":
            inside, block = True, []
            continue
        if inside and line.strip() == "```":
            yield "\n".join(block)
            inside = False
            continue
        if inside:
            block.append(line)


@unittest.skipUnless(all(p.is_file() for p in REAL_STDOUT.values()),
                     "preserved real target stdout fixtures not present")
class RecipeTargetGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.helper = Path(cls._tmp.name) / "gate_target_json.py"
        cls.helper.write_text(extract_helper_source())
        cls.real = {m: p.read_text() for m, p in REAL_STDOUT.items()}
        cls.rec = {m: extract_one_json(t) for m, t in cls.real.items()}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_gate(self, stdout_text, mode):
        p = Path(self._tmp.name) / f"case_{mode}.stdout"
        p.write_text(stdout_text)
        return subprocess.run([sys.executable, str(self.helper), str(p), mode],
                              capture_output=True, text=True)

    def run_rec(self, rec, mode):
        return self.run_gate(json.dumps(rec), mode)

    # --- positives: all four preserved real fixtures pass -------------------
    def test_real_fixtures_pass(self):
        for mode in ("identity", "check", "profile", "timing"):
            r = self.run_gate(self.real[mode], mode)
            self.assertEqual(r.returncode, 0, f"{mode} should pass: {r.stderr}")
            self.assertIn("OK target JSON gate: mode=" + mode, r.stdout)

    def test_profile_fixture_is_ncu_wrapped(self):
        # Guard that the profile positive really exercises wrapped extraction.
        self.assertIn("==PROF==", self.real["profile"])

    # --- helper negatives: one-field mutation each must exit nonzero --------
    def _assert_reject(self, mode, mutate, label):
        rec = json.loads(json.dumps(self.rec[mode]))  # deep copy
        mutate(rec)
        r = self.run_rec(rec, mode)
        self.assertNotEqual(r.returncode, 0,
                            f"[{mode}/{label}] must exit nonzero, got 0")

    def test_timing_invalid_means_rejected(self):
        def set_first(rec, v):
            rec["kernels"][0]["event_mean_us_trials"][0] = v
        for label, v in [("+inf", math.inf), ("-inf", -math.inf), ("nan", math.nan),
                         ("bool", True), ("zero", 0), ("negative", -1.0), ("string", "x")]:
            self._assert_reject("timing", lambda rec, v=v: set_first(rec, v), f"mean-{label}")

    def test_timing_trial_count_rejected(self):
        def drop(rec):
            for k in rec["kernels"]:
                k["event_mean_us_trials"] = k["event_mean_us_trials"][:4]
        def add(rec):
            for k in rec["kernels"]:
                k["event_mean_us_trials"] = k["event_mean_us_trials"] + [1.0]
        self._assert_reject("timing", drop, "missing-trial")
        self._assert_reject("timing", add, "extra-trial")

    def test_timing_repeats_must_be_20(self):
        self._assert_reject("timing", lambda rec: rec.update(event_repeats_per_trial=0),
                            "repeats-0")

    def test_kernel_set_rejected(self):
        cases = {
            "no-kernel": lambda rec: rec.update(kernels=[]),
            "one-kernel": lambda rec: rec.update(kernels=rec["kernels"][:1]),
            "duplicate-kernel": lambda rec: rec.update(
                kernels=[rec["kernels"][0], json.loads(json.dumps(rec["kernels"][0]))]),
            "wrong-kernel-name": lambda rec: rec["kernels"][0].update(name="orin_bogus"),
        }
        for mode in ("check", "profile", "timing"):
            for label, mut in cases.items():
                self._assert_reject(mode, mut, label)

    def test_wrong_correctness_counts_rejected(self):
        for mode in ("check", "profile", "timing"):
            self._assert_reject(
                mode, lambda rec: rec["kernels"][0].update(checked_copy_elements=1048575),
                "wrong-copy-elements")
            self._assert_reject(
                mode, lambda rec: rec["kernels"][0].update(checked_block_sums=4095),
                "wrong-block-sums")

    def test_check_profile_repeats_rejected(self):
        for mode in ("check", "profile"):
            self._assert_reject(mode, lambda rec: rec.pop("event_repeats_per_trial"),
                                "repeats-missing")
            self._assert_reject(mode, lambda rec: rec.update(event_repeats_per_trial=5),
                                "repeats-nonzero")
            # A boolean False must NOT stand in for the integer 0 (False == 0).
            self._assert_reject(mode, lambda rec: rec.update(event_repeats_per_trial=False),
                                "repeats-bool-false")

    def test_check_profile_means_must_be_empty(self):
        for mode in ("check", "profile"):
            self._assert_reject(
                mode, lambda rec: rec["kernels"][0].update(event_mean_us_trials=[1.0]),
                "means-nonempty")
            self._assert_reject(
                mode, lambda rec: rec["kernels"][0].pop("event_mean_us_trials"),
                "means-missing")

    def test_status_and_device_guards_preserved(self):
        self._assert_reject("check", lambda rec: rec.update(status="failed"), "bad-status")
        self._assert_reject(
            "check", lambda rec: rec["device"].update(name="NVIDIA Thor"), "bad-device-name")
        self._assert_reject(
            "check", lambda rec: rec["device"].update(compute_capability=[11, 0]), "bad-cc")

    # --- recipe bash fences are syntactically valid -------------------------
    def test_recipe_bash_fences_syntax(self):
        fences = list(bash_fences())
        self.assertGreaterEqual(len(fences), 3, "expected several bash fences")
        for i, block in enumerate(fences):
            r = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"fence #{i} bash -n failed: {r.stderr}")


if __name__ == "__main__":
    unittest.main()
