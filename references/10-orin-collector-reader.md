# Orin (sm_87) collection + reader split — runnable recipes

Jetson AGX Orin (CC 8.7) needs the collector and the report reader treated as
**two separate tools with two separate versions**. This page documents the
tested pairing and exact, runnable recipes. The recipes are authored for the
root operator to run on the Orin device; the skill builder never runs CUDA, NCU,
or any install. Assign the absolute paths at the top of each block for the host
you run on — nothing here is hardcoded to one campaign.

## The collector / reader split (tested pairing)

| Role | Tool | Version | Notes |
|---|---|---|---|
| Collector (on-device) | `ncu` | **NCU 2025.2.1** (Orin) | On THIS installation the bundled Python extension is x86-64 and does not import on aarch64. |
| Reader (host/controller) | `ncu-report` wheel | **2026.3.1** (aarch64) | Imports `ncu_report` on aarch64 and reads the 2025.2.1-collected report. |

A newer reader is **not** a newer collector. Do not relabel a 2025.2.1-collected
report as 2026.3.1. (Thor's collector is NCU 2026.1.1 — a different campaign's
identity, preserved in `references/03-collection.md`; it is not changed here.)
The incompatible bundled reader above is a property of *this* 2025.2.1 package
layout, not a claim about every NCU 2025.2.1 distribution.

## Operator-assigned paths + a failure-stopping logger

Set every absolute path once; create a fresh run dir with `mktemp -d` and fail if
creation fails. `run_logged` records each command's stdout/stderr/exit and
returns the command's real exit code (never masked by a pipeline), so every step
is gated with `|| exit "$?"`.

```bash
RUN_PARENT=/abs/path/to/profile            # existing operator-assigned parent
SCRIPTS=/abs/path/to/skills/ncu-report-skill/scripts
SRC=/abs/path/to/orin_probe.cu             # reviewed probe source
NVCC=/usr/local/cuda-12.6/bin/nvcc         # CUDA 12.6 toolchain
NCU=/opt/nvidia/nsight-compute/2025.2.1/ncu
READER_PY=/abs/path/to/ncu-reader-venv/bin/python   # aarch64 ncu-report==2026.3.1

OUT=$(mktemp -d "$RUN_PARENT/orin-XXXXXXXX") || exit "$?"
mkdir "$OUT/tmp" "$OUT/reports" || exit "$?"
BIN="$OUT/orin_probe"

run_logged() {                 # run_logged <label> <cmd...>
  local label="$1"; shift
  "$@" > "$OUT/$label.stdout" 2> "$OUT/$label.stderr"
  local rc=$?
  # Fail the step if the required exit receipt cannot be written, rather than
  # masking the write failure behind the command's own status.
  printf '%s\n' "$rc" > "$OUT/$label.exit" || return "$?"
  return "$rc"
}

# One shared gate for the probe's OWN target JSON. Each --mode invocation prints
# exactly one JSON record; under NCU that record is surrounded by NCU's stdout,
# so this extracts the single top-level JSON object and checks the reviewed
# fields for the mode. It validates the TARGET's stdout — which the later reader
# result-gate does not — accepting exactly one record per invocation.
cat > "$OUT/gate_target_json.py" <<'PY'
import json, math, sys
path, mode = sys.argv[1], sys.argv[2]
text = open(path).read()
objs, depth, start = [], 0, None
for i, ch in enumerate(text):
    if ch == '{':
        if depth == 0:
            start = i
        depth += 1
    elif ch == '}' and depth:
        depth -= 1
        if depth == 0:
            try:
                objs.append(json.loads(text[start:i + 1]))
            except Exception:
                pass
assert len(objs) == 1, "expected exactly one target JSON record, got %d" % len(objs)
rec = objs[0]
assert rec.get("mode") == mode, (rec.get("mode"), mode)
dev = rec.get("device", {})
assert dev.get("name") == "Orin" and dev.get("compute_capability") == [8, 7], dev
want = {"identity": "observed", "check": "passed",
        "profile": "passed", "timing": "passed"}[mode]
assert rec.get("status") == want, (rec.get("status"), want)

def finite_pos(x):
    # A real finite, strictly-positive sample: an int/float that is NOT a bool,
    # and not +inf/-inf/NaN. Rejects booleans, infinities, NaN, zero, negatives
    # and non-numbers without normalizing the sample.
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and x > 0)

def int_count(x):
    # An actual integer count; a bool must never stand in for an int (False == 0).
    return isinstance(x, int) and not isinstance(x, bool)

if mode in ("check", "profile", "timing"):
    kernels = rec.get("kernels", [])
    assert sorted(k["name"] for k in kernels) == [
        "orin_cp_async_copy_reduce", "orin_simt_copy_reduce"], kernels
    for k in kernels:
        assert k["checked_copy_elements"] == 1048576 and k["checked_block_sums"] == 4096, k
if mode in ("check", "profile"):
    reps = rec.get("event_repeats_per_trial")
    assert int_count(reps) and reps == 0, reps
    for k in rec["kernels"]:
        assert k["event_mean_us_trials"] == [], (k["name"], k["event_mean_us_trials"])
if mode == "timing":
    assert rec.get("event_repeats_per_trial") == 20, rec.get("event_repeats_per_trial")
    for k in rec["kernels"]:
        means = k["event_mean_us_trials"]
        assert len(means) == 5 and all(finite_pos(m) for m in means), (k["name"], means)
print("OK target JSON gate: mode=" + mode)
PY

gate_target_json() {           # gate_target_json <label> <mode>
  run_logged "gate-$1" python3 "$OUT/gate_target_json.py" "$OUT/$1.stdout" "$2" || exit "$?"
}
```

## Preflight gate (root, before collecting)

A name-only query does not establish idle/compatible. Require all of: the
root-controlled campaign advisory lock held for this run; fresh runtime identity
CC **8.7** / name **Orin** read from the device; and the campaign's actual
process + live-telemetry inspection. Record utilization/compute-process fields
verbatim; when the platform reports `[N/A]`, keep it **unavailable** — never read
`[N/A]` as "idle". Abort on another process holding the GPU or a non-Orin /
non-8.7 identity. Do not change clocks or power. A probe device-attribute read is
not live utilization telemetry — keep the operator telemetry check separate.

## Build, identity, and correctness (run BEFORE profiling)

The reviewed `orin_probe` defaults to `--mode check`; profiling needs
`--mode profile`. Build native `sm_87` (CUDA 12.6), then run `--mode identity`
and `--mode check` with gated receipts, parsing exactly one target JSON record
per run (identity `status=observed`, check `status=passed`, device name `Orin`,
CC `[8,7]`, the exact-copy/block-sum validation and the two expected kernel
names). The historical `reports/device-probe-v1/commands.json` records the
equivalent `-arch=sm_87` compilation.

```bash
run_logged build env TMPDIR="$OUT/tmp" "$NVCC" -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_87,code=sm_87 -Xptxas=-v "$SRC" -o "$BIN" || exit "$?"
run_logged binary-ready test -x "$BIN" || exit "$?"
run_logged identity env TMPDIR="$OUT/tmp" "$BIN" --mode identity || exit "$?"
# Gate the identity target JSON (mode=identity, status=observed, name=Orin,
# cc=[8,7]); the separate process/live-telemetry inspection is still required.
gate_target_json identity identity
run_logged check env TMPDIR="$OUT/tmp" "$BIN" --mode check || exit "$?"
# Gate the check target JSON (mode=check, status=passed, both kernel names,
# exact 1048576/4096 copy+block-sum counts, integer event_repeats_per_trial=0,
# and empty per-kernel event_mean_us_trials — timing is a separate mode).
gate_target_json check check
```

## Independent event timing (separate from NCU)

CUDA-event timing is a **separate** run from any profiled collection — never time
a kernel inside an NCU replay. The reviewed probe (`--mode timing`, no extra
flags) runs five trials of `event_repeats_per_trial = 20` timed launches per
kernel; the gate below requires five finite positive per-trial means for each
kernel.

```bash
run_logged timing env TMPDIR="$OUT/tmp" "$BIN" --mode timing || exit "$?"
gate_target_json timing timing
```

These event means are a bounded synthetic repeated-copy measurement only; they
establish no operator, bandwidth or speedup claim, and are kept out of the NCU
collection below.

## Collection recipe (root runs on Orin; NCU 2025.2.1)

Keep CUDA-event timing in a SEPARATE run (`--mode timing`); never time inside a
profiled run. Use a command-scoped `TMPDIR` that survives `sudo`, and omit
`--force-overwrite` so earlier evidence is never clobbered. `run_logged` captures
NCU's real exit — do not mask it with `echo | tee`.

```bash
report="$OUT/reports/orin.ncu-rep"      # one report; the probe emits two actions
run_logged collect sudo -n env TMPDIR="$OUT/tmp" "$NCU" \
  --profile-from-start off \
  --clock-control none --cache-control all --replay-mode kernel \
  --target-processes all \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats --section InstructionStats \
  --section SourceCounters --section PmSampling --section PmSampling_WarpStates \
  --export "$report" \
  "$BIN" --mode profile || exit "$?"
run_logged report-ready test -s "$report" || exit "$?"
# Gate the probe's own target JSON (mode=profile, status=passed, both kernels,
# exact 1048576/4096 copy+block-sum counts, integer event_repeats_per_trial=0 and
# empty per-kernel event_mean_us_trials), extracted from collect.stdout even
# though NCU wraps it in surrounding output.
gate_target_json collect profile
```

`--clock-control none` leaves clocks as-is: NCU warns results may be inconsistent,
which is expected — this is a correctness/attribution capture, not a timing claim.

## Reader + result gate (controller; aarch64 ncu-report==2026.3.1)

`load_capture()` runs the capture child with `sys.executable` under `-I` unless
another interpreter is passed, so select the compatible reader explicitly — a
module path alone does not establish native-extension/Python-ABI compatibility.
Record the reader version (`$READER_PY -m pip show ncu-report`). Clear any
inherited `PYTHONPATH` / `NCU_REPORT_PYTHONPATH` on the reader commands with
`env -u`: `load_capture` bridges either variable into the isolated capture child,
so selecting the venv interpreter alone does not stop an unintended external SDK
path from reaching the reader. The venv's own `ncu_report` still imports under
`-I` from the interpreter's own prefix. (To intentionally bridge a *separate*
reader SDK, set `NCU_REPORT_PYTHONPATH` explicitly on that one command instead of
inheriting it.) Gate the claimed result on the detached capture's own `status`
(NOT analyzer stdout, which `load_capture` does not forward):

```bash
run_logged analyze env -u PYTHONPATH -u NCU_REPORT_PYTHONPATH TMPDIR="$OUT/tmp" \
  "$READER_PY" -I -B \
  "$SCRIPTS/analyze_reports.py" --run-dir "$OUT" \
  --report "$OUT/reports/orin.ncu-rep" --tag orin --profile orin || exit "$?"

run_logged result-gate env -u PYTHONPATH -u NCU_REPORT_PYTHONPATH TMPDIR="$OUT/tmp" \
  "$READER_PY" -I - \
  "$SCRIPTS" "$OUT" "$OUT/reports/orin.ncu-rep" <<'PY' || exit "$?"
import hashlib, json, sys
scripts, out, report = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, scripts)
from pathlib import Path
from ncu_utils import load_capture, select_actions, read_identity
report = Path(report)
cap = load_capture(report, output_path=Path(out) / "native-capture.json")
assert cap["schema_version"] == 3
assert cap["status"] == "captured_native_report", cap["status"]
assert not cap["structural_errors"]
acts = select_actions(cap)
assert cap["num_actions"] == len(acts) == 2, len(acts)
assert sorted(a["name"] for a in acts) == [
    "orin_cp_async_copy_reduce", "orin_simt_copy_reduce"]
for a in acts:
    ident = read_identity(a)
    assert ident["display_name"] == "Orin" and ident["cc"] == (8, 7), ident
assert cap["report"]["sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
# Each per-action sidecar must confirm the Orin selection.
for a in acts:
    sel = json.loads((Path(out) / "analysis" /
                      f"profile_selection_orin_r{a['range_index']}a{a['action_index']}.json").read_text())
    assert sel["requested_profile"] == "orin" and sel["chosen_profile"] == "orin"
    assert sel["status"] == "confirmed", sel["status"]
print("OK: 2 actions, Orin/(8,7), digest + sidecars confirmed")
PY
```

A missing native reader **fails loudly** (the capture child raises; the parent
raises `CaptureError`). There is no fake/fallback reader in the production path.

> **Profile-selection caveat.** A small fixed *expected-identity* constraint is
> kept per explicit profile — `orin` → name `Orin`/CC 8.7, `thor` → `NVIDIA Thor`/
> CC 11.0, `b200` → `NVIDIA B200` or `B200`/CC 10.0 — and is separate from
> *qualification*. Only Orin is empirically qualified (auto-selects `orin` on a
> readable `Orin`/8.7; auto on any other identity, including Thor/B200, stays
> `generic`, never B200 by default). The expected-identity constraint is used only
> to **reject a readable contradiction**, not to claim a device was measured:
> for an explicit `--profile orin`, `b200`, or `thor`, a single individually
> readable identity that contradicts that profile — a display name recognized as a
> **different** modeled target, or an incompatible CC major/minor, even when the
> partner component is absent/unreadable — yields `chosen_profile=generic`,
> `status=conflict`, and CLI **exit 1**. An explicit, fully qualified `Orin`/8.7
> selection is `confirmed` (exit 0). For b200/thor, a matched name/CC stays
> `unverified` (exit 0). Missing/unreadable or unfamiliar identities without a
> readable contradiction also stay `unverified` (exit 0). For b200/thor,
> `unverified` means "identity not
> established for this profile" (no new hardware qualification), **not** "identity
> checked and found consistent." Read the `profile_selection_*.json` `identity`
> block to see what was actually read.

## Repeatability

This `result-gate` `load_capture` is a separate fresh reader operation, not
another GPU measurement. Reading the same report twice in fresh processes and
comparing actions/ranges is a determinism check only. Exit zero or a non-empty
export alone is NOT a successful capture — every step above is gated by its
recorded exit and the structured JSON assertions.
