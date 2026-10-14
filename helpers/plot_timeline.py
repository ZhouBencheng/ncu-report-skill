#!/usr/bin/env python3
"""ASCII-plot PM sampling timeseries from .ncu-rep files.

PM sampling metrics have per-instance values that form a time-ordered series
across the kernel's execution. Plotting the series reveals tail effects,
pipeline bubbles, and sawtooth patterns that are invisible in aggregate metrics.

The series come from the PmSampling section (part of `--set full`), named
`<UNIT>.<group>.<metric>` in ncu 2026.1, and the PmSampling_WarpStates section
(`--section PmSampling_WarpStates`), named `pmsampling:smsp__warps_issue_stalled_*`.
`--list` prints the series a report has.

ncu collects the series over several replays of the kernel, and each replay has
its own time base. Series of one replay share a time axis: they are plotted
together, over the window where any of them is non-zero, so their shapes line
up. Series of different replays do not line up sample for sample — compare
their shapes, not positions.

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
    vals = [v if v is not None else 0.0 for v in vals]
    n = len(vals)
    if n == 0 or not any(vals):
        return [f"\n{label}: all zero"]

    # Bucket into max_cols columns
    ncols = min(max_cols, n)
    buckets = []
    for c in range(ncols):
        chunk = vals[c * n // ncols:(c + 1) * n // ncols]
        buckets.append(sum(chunk) / len(chunk) if chunk else 0.0)
    mx = max(buckets) or 1.0

    lines = [f"\n{label}", f"  (max={mx:.3g})"]
    for r in range(max_rows, 0, -1):
        threshold = mx * r / max_rows
        row = "".join("#" if b >= threshold else " " for b in buckets)
        lines.append(f"  {threshold:8.2g} | {row}")
    lines.append("  " + " " * 10 + "-" * len(buckets))
    lines.append("  " + " " * 10 + " (time →)")
    return lines


def plot_replays(action, names, rows, cols):
    """Plot `names`, grouped by the replay (time base) each series was sampled in."""
    replays = {}
    for name in names:
        ts, vals = pm_series(action, name)
        key = (ts[0], ts[-1], len(ts)) if ts else name
        replays.setdefault(key, []).append((name, ts, vals))
    out = []
    for k, group in enumerate(replays.values()):
        ts = group[0][1]
        n = len(group[0][2])
        active = [i for i in range(n) if any(v[i] for _, _, v in group)]
        lo, hi = (active[0], active[-1] + 1) if active else (0, n)
        span = f", {(ts[hi - 1] - ts[lo]) / 1e3:.1f} us" if ts and hi - lo > 1 else ""
        interval = f", ~{(ts[-1] - ts[0]) / (len(ts) - 1) / 1e3:.3g} us/sample" if ts and len(ts) > 1 else ""
        out.append(f"\n----- replay {k}: samples {lo}..{hi} of {n}{span}{interval}; "
                   f"these {len(group)} series share this time axis -----")
        for name, _, vals in group:
            out.extend(ascii_plot(vals[lo:hi], name, rows, cols))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--metric", type=str, action="append", default=None,
                    help="Regex on series names (repeatable); replaces the default list.")
    ap.add_argument("--kernel", type=str, default=None,
                    help="Regex on the demangled kernel name: keep only matching kernels.")
    ap.add_argument("--list", action="store_true", help="Print the series each report has and exit.")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--cols", type=int, default=80)
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")

    patterns = [re.compile(p) for p in (args.metric or DEFAULT_PATTERNS)]
    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        _report, acts = select_actions(rep, args.kernel)
        for action, t in zip(acts, action_tags(tag, acts)):
            names = pm_series_names(action)
            if args.list:
                print(f"[{t}] {len(names)} series")
                print("\n".join(f"  {n}" for n in names))
                continue
            if not names:
                print(f"[{t}] no PM sampling series: collect with --set full "
                      f"(and --section PmSampling_WarpStates for stall timelines)", file=sys.stderr)
                continue
            out_lines = [f"{'=' * 60}\n{t}: {rep.name}\n{kernel_name(action, 'function')}\n{'=' * 60}"]
            chosen = [n for p in patterns for n in names if p.search(n)]
            out_lines.extend(plot_replays(action, list(dict.fromkeys(chosen)), args.rows, args.cols))
            out_path = analysis_dir / f"pm_timeline_{t}.txt"
            out_path.write_text("\n".join(out_lines) + "\n")
            print(f"[{t}] -> {out_path}")


if __name__ == "__main__":
    main()
