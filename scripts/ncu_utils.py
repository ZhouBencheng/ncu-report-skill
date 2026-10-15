"""Shared helpers for consuming detached Nsight Compute captures.

This module imports and runs on a CPU-only host: it does NOT import the native
``ncu_report`` module. Each report is read once, in a fresh child process, by
``ncu_capture.py`` (the single native-import site); this module launches that
child, loads the detached JSON record, and exposes accessors that aggregate only
from the detached data — never from a live native object.

Typical use:
    from ncu_utils import load_capture, select_actions, metric_value
    capture = load_capture("report.ncu-rep")
    for action in select_actions(capture):      # every range/action, not just 0
        ...

The native capture child needs the selected report READER's module on its import
path. The reader and the COLLECTOR are separate tools with separate versions:

  * On the Orin controller the reader is the aarch64 ``ncu-report==2026.3.1``
    wheel (its venv ``site-packages`` carries ``ncu_report``); the report it
    reads was COLLECTED on-device by Orin's NCU **2025.2.1**, whose bundled
    x86-64 Python extension does not import on aarch64. A newer reader is not a
    newer collector — never relabel a 2025.2.1-collected report as 2026.3.1.
  * On Thor the collector is NCU **2026.1.1** and its
    ``extras/python`` provides the reader; that is a different campaign's
    identity and is preserved unchanged (see ``references/03-collection.md``).

Point ``PYTHONPATH`` or ``NCU_REPORT_PYTHONPATH`` at the chosen reader's module
directory for the root-run regression. CPU tests inject a fake ``ncu_report`` and
call ``ncu_capture.capture`` directly instead; there is no fake/fallback reader
in the production path — a missing native reader fails loudly.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
CAPTURE_SCRIPT = HERE / "ncu_capture.py"

# Single source of truth for the sentinel keys the capture emits.
from ncu_capture import NATIVE_ERROR_KEY, NONFINITE_KEY  # noqa: E402


class CaptureError(RuntimeError):
    """The detached capture child failed; raised loudly rather than swallowed."""


class ActionSelectorError(ValueError):
    """An action selector was ambiguous or matched no action (never silently 0)."""


# --- Producing / loading a detached capture ----------------------------------

def load_capture(report_path, *, python=None, output_path=None, env=None):
    """Run the capture child in a fresh process and return the detached dict.

    The native ``ncu_report`` is imported only inside that child; this parent
    process never imports it. Raises ``CaptureError`` on a non-zero exit or a
    structural capture failure.
    """
    report_path = Path(report_path)
    if not report_path.is_file():
        raise CaptureError(f"report does not exist: {report_path}")
    python = python or sys.executable
    tmp = None
    try:
        if output_path is None:
            tmp = tempfile.mkdtemp(prefix="ncu_capture_")
            output_path = Path(tmp) / "capture.json"
        output_path = Path(output_path)
        # -I isolates the child so a planted module in the report directory cannot
        # be imported. Isolated Python ignores PYTHONPATH, so bridge the selected
        # SDK path into NCU_REPORT_PYTHONPATH, which the child splits on
        # os.pathsep and re-inserts (absolute existing dirs only). This keeps the
        # documented `PYTHONPATH=<extras/python>` setup working, including a
        # multi-directory list.
        child_env = dict(os.environ if env is None else env)
        sdk = child_env.get("NCU_REPORT_PYTHONPATH") or child_env.get("PYTHONPATH")
        if sdk:
            child_env["NCU_REPORT_PYTHONPATH"] = sdk
        proc = subprocess.run(
            [python, "-I", str(CAPTURE_SCRIPT), "--report", str(report_path),
             "--output", str(output_path)],
            capture_output=True, text=True, env=child_env,
        )
        capture = None
        if output_path.is_file():
            try:
                capture = json.loads(output_path.read_text())
            except json.JSONDecodeError:
                capture = None
        # A non-zero child exit includes the partial case (some range/action
        # failed to construct): the child still wrote a detached JSON preserving
        # the good actions (status "partial_native_report") for direct CLI/file
        # diagnostic inspection. This Python consumer is deliberately strict
        # fail-loud — it raises rather than return a subset of actions, so a
        # caller never silently aggregates a partial read.
        if proc.returncode != 0:
            # Surface the structured error the child already archived, not the
            # (empty) stderr of a child that reports via its JSON output.
            detail = ""
            if capture is not None:
                detail = str(capture.get(NATIVE_ERROR_KEY) or capture.get("status") or "")
            raise CaptureError(
                f"capture child failed (exit {proc.returncode}) for {report_path}: "
                f"{detail or proc.stderr.strip() or '(no detail)'}")
        if capture is None:
            raise CaptureError(
                f"capture produced no readable output for {report_path}: "
                f"{proc.stderr.strip()}")
        # No separate `status == "capture_failed"` guard is needed: the child sets
        # that status only in its except branch, which always exits non-zero, so
        # the returncode check above already raised (surfacing the status/error).
        return capture
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


# --- Selecting actions (enumerate-all or explicit, never silently 0) ---------

def select_actions(capture, selector=None):
    """Return the selected actions from a capture.

    ``selector=None`` enumerates EVERY range/action (the default — no
    first-action-only collapse). A string ``"rXaY"`` selects one range/action; an
    int selects a flat index into the action list. A selector that matches no
    action, or is ambiguous/unrecognized, raises ``ActionSelectorError`` — it is
    never resolved to action 0.
    """
    actions = capture.get("actions") or []
    if selector is None:
        return list(actions)
    if isinstance(selector, int):
        if 0 <= selector < len(actions):
            return [actions[selector]]
        raise ActionSelectorError(f"action index {selector} out of range (have {len(actions)})")
    text = str(selector).strip()
    if text.isdigit():
        return select_actions(capture, int(text))
    import re
    m = re.fullmatch(r"r(\d+)a(\d+)", text)
    if not m:
        raise ActionSelectorError(f"unrecognized action selector {selector!r} (use 'rXaY' or an index)")
    ri, ai = int(m.group(1)), int(m.group(2))
    matches = [a for a in actions if a.get("range_index") == ri and a.get("action_index") == ai]
    if not matches:
        raise ActionSelectorError(f"no action matches selector {selector!r}")
    if len(matches) > 1:
        raise ActionSelectorError(f"selector {selector!r} is ambiguous ({len(matches)} matches)")
    return matches


def action_label(action):
    return f"r{action.get('range_index')}a{action.get('action_index')}"


def action_slug(tag, action):
    """A per-action output slug combining the report tag and the action label."""
    return f"{tag}_{action_label(action)}"


def action_name(action):
    name = action.get("name")
    return name if isinstance(name, str) else action_label(action)


# --- Classifying the five distinguishable value states -----------------------

ABSENT = "absent"
READ_ERROR = "read-error"
NONFINITE_NAN = "nan"
NONFINITE_POS_INF = "+inf"
NONFINITE_NEG_INF = "-inf"
TRUE_ZERO = "true-zero"
VALUE = "value"


def classify_value(detached):
    """Classify a detached value into one of the distinguishable states.

    Returns one of: 'absent', 'read-error', 'nan', '+inf', '-inf', 'true-zero',
    'value'. These stay distinct and JSON-safe; numeric zero is never conflated
    with a missing or failed read.
    """
    if detached is None:
        return ABSENT
    if isinstance(detached, dict):
        if NATIVE_ERROR_KEY in detached:
            return READ_ERROR
        if NONFINITE_KEY in detached:
            token = str(detached[NONFINITE_KEY]).lower()  # repr of a non-finite float
            if "nan" in token:
                return NONFINITE_NAN
            return NONFINITE_NEG_INF if token.startswith("-") else NONFINITE_POS_INF
        return VALUE
    if isinstance(detached, bool):
        return VALUE
    if isinstance(detached, (int, float)) and detached == 0:
        return TRUE_ZERO
    return VALUE


def is_numeric_state(state):
    """True for a real numeric sample (an ordinary value or a true zero)."""
    return state in (VALUE, TRUE_ZERO)


def metric_record(action, name):
    """Return the detached metric record for ``name``, or an absent marker."""
    metrics = action.get("metrics") or {}
    rec = metrics.get(name)
    if rec is None:
        return {"name": name, "present": False}
    return rec


def metric_present(action, name):
    return bool(metric_record(action, name).get("present"))


def metric_value(action, name):
    """Return the detached scalar value of a metric (may be a sentinel dict).

    A metric that is listed but failed to read returns its read-error sentinel
    (so it classifies as 'read-error', not 'absent'); a truly absent metric
    returns None. Callers distinguish states via ``classify_value``.
    """
    rec = metric_record(action, name)
    if rec.get(NATIVE_ERROR_KEY) is not None:
        return {NATIVE_ERROR_KEY: rec[NATIVE_ERROR_KEY]}
    if not rec.get("present"):
        return None
    return rec.get("value")


def metric_display(action, name):
    """Availability-aware rendering of a scalar metric for human output.

    Keeps absent, read-error, and scalar-unavailable (native ``has_value()`` is
    False) distinct from a true numeric zero. Availability applies to the SCALAR
    only; per-instance availability is not exposed by the native API.
    """
    rec = metric_record(action, name)
    if rec.get(NATIVE_ERROR_KEY) is not None:
        return "<read-error>"
    if not rec.get("present"):
        return "<absent>"
    state = classify_value(rec.get("value"))
    if state == READ_ERROR:
        return "<read-error>"
    # `has_value` is a separate axis and may itself have read-errored. A failed
    # availability read must not fall through to rendering the (possibly zero)
    # value as if availability were known.
    has_value = rec.get("has_value")
    if classify_value(has_value) == READ_ERROR:
        return "<read-error>"
    if has_value is False:
        return "<unavailable>"
    if is_numeric_state(state):
        return rec.get("value")
    return f"<{state}>"


def metric_has_value(action, name):
    """Return the native scalar ``has_value()`` result, kept SEPARATE from value."""
    rec = metric_record(action, name)
    return rec.get("has_value") if rec.get("present") else None


def metric_unit(action, name):
    rec = metric_record(action, name)
    return rec.get("unit") if rec.get("present") else None


def unit_is_known(unit):
    """A native unit is a comparable, known unit only if it is a string.

    A readable empty string ``""`` is a known (dimensionless) unit; ``None``
    (missing) and a native-error dict (unreadable) are NOT known — they must
    never be treated as directly comparable, and never be inferred into a
    physical unit.
    """
    return isinstance(unit, str)


def metric_unit_display(action, name):
    """Bracketed native-unit annotation for the availability-aware curated view.

    The unit is a SEPARATE axis from the value state; this annotation never
    changes the value-state token (true zero stays distinct from absent /
    unavailable / read-error / nonfinite). Rules, with no unit inferred from the
    metric name:
      * a readable unit string is shown verbatim (``[ns]``);
      * a readable *empty* string stays distinguishable from a missing unit
        (``[unit: empty]`` vs ``[unit: unknown]``);
      * an unreadable (native-error) unit renders ``[unit: read-error]``;
      * an absent metric gets no annotation (the ``<absent>`` token already says
        there is nothing to annotate).
    """
    rec = metric_record(action, name)
    if not rec.get("present"):
        return ""
    unit = rec.get("unit")
    if isinstance(unit, dict) and NATIVE_ERROR_KEY in unit:
        return "[unit: read-error]"
    if unit is None:
        return "[unit: unknown]"
    if isinstance(unit, str):
        return "[unit: empty]" if unit == "" else f"[{unit}]"
    return "[unit: unknown]"


def instance_values(action, name):
    """Return the per-instance detached values list, or None if not exposed."""
    rec = metric_record(action, name)
    instances = rec.get("instances")
    if not isinstance(instances, list):
        return None
    return [inst.get("value") for inst in instances]


# --- Rule results (from the detached capture; no duration-speedup inference) --

def rule_results(action):
    rr = action.get("rule_results")
    return rr if isinstance(rr, list) else []


def rule_speedups(action):
    """Return (est_speedup_pct, rule_name, message) from the native rule engine.

    These are the rule engine's own estimates, not a speedup derived from NCU
    replay duration. Missing estimates become 0.0.
    """
    out = []
    for rr in rule_results(action):
        if not isinstance(rr, dict):
            continue
        est = rr.get("estimated_speedup_pct")
        try:
            est = float(est) if est is not None else 0.0
        except (TypeError, ValueError):
            est = 0.0
        out.append((est, rr.get("rule_name", "?"), rr.get("message_for_display", "?")))
    out.sort(key=lambda x: -x[0])
    return out


# --- Curated metric sets -----------------------------------------------------
#
# Known-meaningful names on B200 / sm_100 with Nsight Compute 2026.x. Other GPU
# generations and ncu releases may need alternate names — always verify against
# an action's metric_names. See ../references/08-b200-metric-names.md.
B200_KEY_METRICS = [
    "launch__grid_size",
    "launch__block_size",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__block_dim_x",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__thread_count",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "device__attribute_multiprocessor_count",
    "device__attribute_max_warps_per_multiprocessor",
    "gpu__time_duration.sum",
    "smsp__cycles_active.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.max.per_cycle_active",
    "sm__warps_active.min.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_eligible.max.per_cycle_active",
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg",
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_misc_per_issue_active.ratio",
    "smsp__pcsamp_sample_count",
]


# --- Per-device curated metric profiles (fixed, data-only) -------------------
#
# `metrics_key_*` views select ONE of these lists by the analyzer's --profile.
# The raw `metrics_all_*` archive is identical regardless of profile. Rules:
#   * `b200` IS the exported ``B200_KEY_METRICS`` constant itself (not a copy),
#     so the preserved constant and the historical B200 view are unchanged.
#   * `orin` is the evidence-verified curated scalar set — every name is present
#     with a finite numeric scalar and ``has_value=true`` in both actions of the
#     real Orin report (see ../references + inputs/orin-native-evidence-review.md).
#     No metric is invented; absent B200-only names are deliberately excluded.
#   * `generic` / `thor` are documented subsets of known NCU names (all drawn
#     from ``B200_KEY_METRICS``). `thor` reuses the Blackwell-family names Thor
#     shares; it is a curated selection, not a claim verified against a Thor
#     report. No name here is invented.

GENERIC_KEY_METRICS = [
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__waves_per_multiprocessor",
    "launch__thread_count",
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "device__attribute_multiprocessor_count",
]

ORIN_KEY_METRICS = [
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__registers_per_thread_allocated",
    "launch__shared_mem_per_block",
    "launch__waves_per_multiprocessor",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "profiler__replayer_passes",
]

THOR_KEY_METRICS = [
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__waves_per_multiprocessor",
    "launch__thread_count",
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "device__attribute_multiprocessor_count",
]

# Fixed profile -> metric-list mapping. `b200` references the constant itself.
PROFILE_METRICS = {
    "generic": GENERIC_KEY_METRICS,
    "b200": B200_KEY_METRICS,
    "thor": THOR_KEY_METRICS,
    "orin": ORIN_KEY_METRICS,
}
PROFILE_CHOICES = ("auto", "generic", "b200", "thor", "orin")

# Typed, readable identity fields used for auto/qualified selection.
IDENTITY_DISPLAY_NAME = "device__attribute_display_name"
IDENTITY_CC_MAJOR = "device__attribute_compute_capability_major"
IDENTITY_CC_MINOR = "device__attribute_compute_capability_minor"

# The ONLY positively-qualified device identity. A profile is auto-selected or
# confirmed only on an exact match; everything else degrades to `generic` (never
# B200 by default). Orin is (display_name "Orin", CC (8, 7)).
ORIN_IDENTITY = ("Orin", (8, 7))
QUALIFIED_IDENTITIES = {ORIN_IDENTITY: "orin"}

# A small FIXED expected-identity constraint per explicit device profile, kept
# SEPARATE from the qualification above. It is used ONLY to reject a readable
# contradiction for an explicit `--profile`; it does NOT qualify a profile for
# auto-selection and does NOT claim the device was measured. Names are matched
# EXACTLY (both existing B200 fixture spellings are kept) and CC is the exact
# (major, minor) expected for that SKU — no SKU is inferred from CC alone, a
# substring, a filename, or the request. Only Orin is additionally qualified
# (QUALIFIED_IDENTITIES, above); thor's 11.0 and b200's 10.0 are the expected
# values from the reviewed synthetic Thor capture and the frozen Blackwell wiki +
# NVIDIA CC table, NOT a report collected in this campaign.
EXPECTED_IDENTITIES = {
    "orin": {"names": ("Orin",), "cc": ORIN_IDENTITY[1]},
    "thor": {"names": ("NVIDIA Thor",), "cc": (11, 0)},
    "b200": {"names": ("NVIDIA B200", "B200"), "cc": (10, 0)},
}

# Every display name recognized as naming a modeled target. A readable name in
# this set that is not one of the requested profile's own names contradicts it
# (it names a DIFFERENT modeled target); a readable name ABSENT from this set is
# merely unfamiliar and cannot, on its own, establish a contradiction.
RECOGNIZED_DEVICE_NAMES = {
    name: profile
    for profile, spec in EXPECTED_IDENTITIES.items()
    for name in spec["names"]
}


def _readable_scalar(action, name):
    """Return (value, readable). A field is readable only if the metric is
    present, its scalar availability is not False (and not itself a read error),
    and the value is a real scalar (not read-error / non-finite). A stale value
    behind ``has_value=False``, or an unreadable availability flag, is NOT
    readable — it must never establish device identity.
    """
    rec = metric_record(action, name)
    if not rec.get("present"):
        return None, False
    if rec.get(NATIVE_ERROR_KEY) is not None:
        return None, False
    has_value = rec.get("has_value")
    if has_value is False or classify_value(has_value) == READ_ERROR:
        return None, False
    val = rec.get("value")
    if classify_value(val) not in (VALUE, TRUE_ZERO):
        return None, False
    return val, True


def _integral_cc_component(action, name):
    """Return a CC component as int only if readable and exactly integral. A
    Boolean, or a fractional float, is rejected — never truncated into a match."""
    val, ok = _readable_scalar(action, name)
    if not ok or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return None


def read_identity(action):
    """Read per-action device identity from typed, readable, available fields.

    Each field is validated independently: an absent / read-error / non-finite /
    scalar-unavailable (``has_value`` False) field is unreadable (None) and never
    establishes identity. CC components are kept individually so a single
    readable component can still contradict a requested target when its partner
    is absent; ``cc`` is the (major, minor) pair only when BOTH are readable.
    """
    disp, disp_ok = _readable_scalar(action, IDENTITY_DISPLAY_NAME)
    display_name = disp if (disp_ok and isinstance(disp, str)) else None
    cc_major = _integral_cc_component(action, IDENTITY_CC_MAJOR)
    cc_minor = _integral_cc_component(action, IDENTITY_CC_MINOR)
    cc = (cc_major, cc_minor) if (cc_major is not None and cc_minor is not None) else None
    return {
        "display_name": display_name,
        "cc": cc,
        "cc_major": cc_major,
        "cc_minor": cc_minor,
        "display_state": classify_value(metric_value(action, IDENTITY_DISPLAY_NAME)),
        "cc_major_state": classify_value(metric_value(action, IDENTITY_CC_MAJOR)),
        "cc_minor_state": classify_value(metric_value(action, IDENTITY_CC_MINOR)),
    }


def _qualified_profile(identity):
    """The profile a fully-readable identity positively qualifies, or None."""
    if identity["display_name"] is not None and identity["cc"] is not None:
        return QUALIFIED_IDENTITIES.get((identity["display_name"], tuple(identity["cc"])))
    return None


def _expected_contradictions(requested, identity):
    """Reasons a readable identity contradicts an explicit profile's fixed
    expected identity (empty list == no contradiction).

    A contradiction is any INDIVIDUALLY readable component that disagrees, even
    when its partner component is absent/unreadable:
      * a display name recognized as a DIFFERENT modeled target (a readable but
        UNFAMILIAR, unrecognized name is NOT, by itself, a contradiction), or
      * a CC major or minor that differs from the profile's expected value.
    No SKU is inferred from CC alone, a substring, a filename, or the request.
    """
    spec = EXPECTED_IDENTITIES[requested]
    names = spec["names"]
    want_major, want_minor = spec["cc"]
    reasons = []
    disp = identity["display_name"]
    if disp is not None and disp not in names and disp in RECOGNIZED_DEVICE_NAMES:
        reasons.append(
            f"display name {disp!r} names a different modeled target "
            f"({RECOGNIZED_DEVICE_NAMES[disp]!r}), not {requested!r}")
    if identity["cc_major"] is not None and identity["cc_major"] != want_major:
        reasons.append(f"CC major {identity['cc_major']} != expected {want_major}")
    if identity["cc_minor"] is not None and identity["cc_minor"] != want_minor:
        reasons.append(f"CC minor {identity['cc_minor']} != expected {want_minor}")
    return reasons


def select_profile(requested, identity):
    """Resolve (chosen_profile, status, reason) from a requested profile + identity.

    Statuses: 'auto-confirmed', 'auto-generic', 'confirmed', 'unverified',
    'conflict'. A readable identity that contradicts an explicit device profile's
    fixed expected identity yields 'conflict' with chosen 'generic' (the caller
    must exit nonzero) — a SINGLE individually readable recognized-other-target
    name, or an incompatible CC major/minor, is sufficient, even when the partner
    component is absent/unreadable. Only Orin is empirically qualified: explicit
    'orin' on a readable Orin/(8,7) is 'confirmed', while explicit 'b200'/'thor'
    carry no new qualification and settle at 'unverified' unless contradicted. An
    unknown/unestablished (or merely unfamiliar, noncontradictory) identity
    degrades to 'generic' under auto and leaves an explicit device selection
    'unverified' — it NEVER falls through to B200.
    """
    qualified = _qualified_profile(identity)
    ident_txt = (f"display={identity['display_name']!r}, cc={identity['cc']}, "
                 f"cc_major={identity['cc_major']}, cc_minor={identity['cc_minor']}")

    if requested == "auto":
        if qualified:
            return qualified, "auto-confirmed", f"identity ({ident_txt}) qualifies '{qualified}'"
        return "generic", "auto-generic", f"no qualified identity ({ident_txt}); generic (never B200 by default)"

    if requested == "generic":
        return "generic", "confirmed", "generic profile explicitly requested"

    # Explicit device profile: 'orin', 'thor', or 'b200'. Only Orin is modeled as
    # qualified (QUALIFIED_IDENTITIES is Orin-only), so only '--profile orin' can
    # reach 'confirmed'. Every explicit profile is checked against its FIXED
    # expected identity (EXPECTED_IDENTITIES): a readable recognized-other-target
    # name or an incompatible CC major/minor is a 'conflict' -> 'generic' (exit
    # nonzero); otherwise it stays with the requested profile as 'unverified'.
    if qualified == requested:                      # only reachable for 'orin'
        return requested, "confirmed", f"identity ({ident_txt}) qualifies {requested!r}"
    contradictions = _expected_contradictions(requested, identity)
    if contradictions:
        return "generic", "conflict", (
            f"explicit profile {requested!r} conflicts with a readable identity "
            f"({ident_txt}): " + "; ".join(contradictions))
    return requested, "unverified", (
        f"explicit profile {requested!r} selected but identity is not "
        f"established ({ident_txt})")
