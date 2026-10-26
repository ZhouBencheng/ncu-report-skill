#!/usr/bin/env python3
"""ASCII-plot PM sampling timeseries and warp-state timelines from ncu reports.

PM sampling metrics have per-instance values that form a time-ordered series
across the kernel's execution. Plotting the series reveals tail effects,
pipeline bubbles, and sawtooth patterns that are invisible in aggregate metrics.

The series come from the PmSampling section (part of `--set full`), named
`<UNIT>.<group>.<metric>` (e.g. `TPC.TriageCompute.sm__cycles_active...`);
`--list` prints the series a report has and its number of timed warp samples. With ncu_report 2026.3+ each series is
aligned to the workload's timestamps.

ncu collects the series over replays of the kernel, and each replay has its own
time base. Series of one replay share a time axis: they are plotted together,
over the window where any of them is non-zero, so their shapes line up. Series
of different replays do not line up sample for sample — compare their shapes,
not positions.

The stall reasons over time come from the timed warp samples the same
section holds: the share of samples in each time
bucket per reason, over the span the kernel had warps resident. (Reports of ncu
before 2026.3 may also have `pmsampling:smsp__warps_issue_stalled_*` series, from
the PmSampling_WarpStates section; they are plotted as PM series.)

Produces `<run-dir>/analysis/pm_timeline_<tag>.txt` per profiled kernel.

Usage:
    python3 plot_timeline.py --run-dir "$PROFILE_RUN_DIR" \\
            --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag> [--list]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    action_tags, kernel_name, pm_series, pm_series_names, select_actions,
    timed_warp_samples, warp_state_timeline,
)


# Default series, as patterns: SM activity, issue rate, tensor pipe, memory
# throughputs, and the stall-reason timelines. Each pattern plots every series it
# matches; `--metric` replaces the list.
DEFAULT_PATTERNS = [
    r"\.sm__cycles_active\.avg",
    r"\.sm__inst_executed_realtime\.avg\.per_cycle_active$",
    r"\.sm__pipe_tensor_cycles_active_realtime\.",
    r"\.dram__throughput\.",
    r"\.lts__throughput\.",
    r"\.l1tex__throughput\.",
    r"^pmsampling:smsp__warps_issue_stalled_(long_scoreboard|short_scoreboard|wait|barrier|"
    r"math_pipe_throttle|mio_throttle|lg_throttle|sleeping)\.",
]


def ascii_plot(vals, label, max_rows=20, max_cols=80):
    """Return list of ASCII strings rendering the timeseries (already windowed)."""
    n = len(vals)
    if not n:
        return [f"\n{label}: no samples"]
    known = [v for v in vals if v is not None]
    if not known:
        return [f"\n{label}: all samples unavailable"]
    if len(known) == n and not any(known):
        return [f"\n{label}: all zero"]

    ncols = min(max_cols, n)
    buckets = []
    for c in range(ncols):
        chunk = vals[c * n // ncols:(c + 1) * n // ncols]
        buckets.append(None if any(v is None for v in chunk) else sum(chunk) / len(chunk))
    mx = max((v for v in buckets if v is not None), default=0) or 1.0
    lines = [f"\n{label}", f"  (max={mx:.3g}; ? = unavailable sample bucket)"]
    for r in range(max_rows, 0, -1):
        threshold = mx * r / max_rows
        row = "".join("?" if b is None else "#" if b >= threshold else " " for b in buckets)
        lines.append(f"  {threshold:8.2g} | {row}")
    lines.append("  " + " " * 10 + "-" * len(buckets))
    lines.append("  " + " " * 10 + " (time →)")
    return lines


def _active_samples(group, n):
    """Sample indices where the kernel runs: SM activity above 1% of its peak when the group has
    it (memory series carry background noise for the whole capture), else where any series is
    above 1% of its own peak. Unknown samples are retained because they cannot be classified as
    inactive measured zeros."""
    sm = [v for name, _, v in group if ".sm__cycles_active." in name]
    if sm:
        if any(x is None for x in sm[0]):
            return list(range(n))
        peak = max(x or 0 for x in sm[0]) or 1
        return [i for i in range(n) if (sm[0][i] or 0) > 0.01 * peak]
    peaks = [max(x or 0 for x in v) for _, _, v in group]
    return [i for i in range(n)
            if any(v[i] is None or (pk and v[i] > 0.01 * pk)
                   for (_, _, v), pk in zip(group, peaks))]


def plot_replays(action, names, rows, cols):
    """Plot `names`, grouped by the replay (time base) each series was sampled in."""
    replays = {}
    for name in names:
        series = pm_series(action, name)
        if series is None:
            continue
        ts, vals = series
        key = (ts[0], ts[-1], len(ts)) if ts else name
        replays.setdefault(key, []).append((name, ts, vals))
    out = []
    for k, group in enumerate(replays.values()):
        ts = group[0][1]
        n = len(group[0][2])
        active = _active_samples(group, n)
        lo, hi = (active[0], active[-1] + 1) if active else (0, n)
        span = f", {(ts[hi - 1] - ts[lo]) / 1e3:.1f} us" if ts and hi - lo > 1 else ""
        interval = f", ~{(ts[-1] - ts[0]) / (len(ts) - 1) / 1e3:.3g} us/sample" if ts and len(ts) > 1 else ""
        out.append(f"\n----- replay {k}: samples {lo}..{hi} of {n}{span}{interval}; "
                   f"these {len(group)} series share this time axis -----")
        for name, _, vals in group:
            out.extend(ascii_plot(vals[lo:hi], name, rows, cols))
    return out


def plot_warp_states(action, rows, cols, top):
    """The `top` most sampled stall reasons over the kernel's run, from its timed warp samples."""
    result = warp_state_timeline(action, cols)
    if result is None:
        return []
    span, shares = result
    order = sorted(shares, key=lambda r: -sum(v for v in shares[r] if v is not None))[:top]
    out = [f"\n----- warp states over the kernel: share of the timed warp samples per reason, "
           f"{span / 1e3:.1f} us (its own time axis) -----"]
    for reason in order:
        out.extend(ascii_plot(shares[reason], f"stall reason: {reason}", rows, cols))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--metric", type=str, action="append", default=None,
                    help="Regex on series names (repeatable); replaces the default list.")
    ap.add_argument("--kernel", type=str, default=None,
                    help="Regex on the kernel's function or demangled name: keep only matching kernels.")
    ap.add_argument("--list", action="store_true", help="Print the series and timed warp sample count of each report and exit.")
    ap.add_argument("--stall-reasons", type=int, default=6,
                    help="How many of the most sampled stall reasons to plot over time (0: none).")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--cols", type=int, default=80)
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    if len(set(args.tag)) != len(args.tag):
        ap.error("each --tag must be different: outputs are named after it")

    patterns = [re.compile(p) for p in (args.metric or DEFAULT_PATTERNS)]
    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        _report, acts = select_actions(rep, args.kernel)
        if not acts:
            print(f"[skip] {rep}: no kernel matches {args.kernel!r}", file=sys.stderr)
            continue
        for action, t in zip(acts, action_tags(tag, acts)):
            names = pm_series_names(action)
            if args.list:
                samples = timed_warp_samples(action)
                sample_count = "unavailable" if samples is None else str(len(samples))
                print(f"[{t}] {len(names)} series, {sample_count} timed warp samples")
                print("\n".join(f"  {n}" for n in names))
                continue
            warp_states = plot_warp_states(action, args.rows, args.cols, args.stall_reasons) \
                if args.stall_reasons else []
            if not names and not warp_states:
                print(f"[{t}] PM timeline unavailable; inspect supported sections and collection coverage. Static metrics remain usable",
                      file=sys.stderr)
                continue
            dur = action["gpu__time_duration.sum"].value() if "gpu__time_duration.sum" in action.metric_names() else None
            dur = f" ({dur / 1e3:.1f} us, gpu__time_duration)" if dur else ""
            out_lines = [f"{'=' * 60}\n{t}: {rep.name}\n{kernel_name(action, 'function')}{dur}\n{'=' * 60}"]
            chosen = [n for p in patterns for n in names if p.search(n)]
            out_lines.extend(plot_replays(action, list(dict.fromkeys(chosen)), args.rows, args.cols))
            out_lines.extend(warp_states)
            out_path = analysis_dir / f"pm_timeline_{t}.txt"
            out_path.write_text("\n".join(out_lines) + "\n")
            print(f"[{t}] -> {out_path}")


if __name__ == "__main__":
    main()
