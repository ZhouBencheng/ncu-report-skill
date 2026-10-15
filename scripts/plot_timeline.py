#!/usr/bin/env python3
"""ASCII-plot PM / warp sampling timeseries from .ncu-rep files.

PM/warp sampling metrics (names prefixed ``pmsampling:`` / ``warpsampling:``)
carry per-instance values forming a time-ordered series across the kernel's
execution. This CLI consumes the detached capture (read once per report in a
fresh process) and plots every such series for EVERY range/action.

Non-finite, missing and failed-read samples are counted and reported rather than
silently coerced to zero, so a true zero stays distinct from a gap.

Produces ``<run-dir>/analysis/pm_timeline_plots.txt``.

Usage:
    python3 plot_timeline.py --run-dir profile/myrun \\
            --report profile/myrun/reports/full_<tag>.ncu-rep --tag <tag>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    action_label, classify_value, instance_values, is_numeric_state, load_capture,
    select_actions,
)


def numeric_series(values):
    """Split detached per-instance values into a numeric series + state counts.

    Returns (series, gaps) where ``series`` keeps numeric samples (including true
    zeros) in order with non-numeric samples represented as None, and ``gaps`` is
    a dict counting the distinguishable non-numeric states so they are reported,
    never silently zeroed.
    """
    series = []
    gaps = {}
    for v in values:
        state = classify_value(v)
        if is_numeric_state(state) and isinstance(v, (int, float)) and not isinstance(v, bool):
            series.append(float(v))
        else:
            series.append(None)
            gaps[state] = gaps.get(state, 0) + 1
    return series, gaps


def ascii_plot(values, label, max_rows=20, max_cols=80):
    if not values:
        return [f"{label}: no instances"]
    series, gaps = numeric_series(values)
    gap_note = (" gaps=" + ",".join(f"{k}:{v}" for k, v in sorted(gaps.items()))) if gaps else ""
    finite = [v for v in series if v is not None]
    if not finite:
        return [f"{label}: no numeric samples ({len(series)} instances;{gap_note or ' all absent'})"]

    # Trim leading/trailing zero-or-gap region for a compact active window.
    def active(v):
        return v is not None and v > 0
    lead = 0
    for v in series:
        if active(v):
            break
        lead += 1
    trail = 0
    for v in reversed(series):
        if active(v):
            break
        trail += 1
    window = series[lead:len(series) - trail] if trail else series[lead:]
    n = len(window)
    if n == 0:
        return [f"{label}: all zero/gap ({len(series)} instances;{gap_note})"]

    ncols = min(max_cols, n)
    # Partition the FULL window into ncols buckets with proportional boundaries,
    # so the final remainder samples are never dropped. A bucket with no numeric
    # sample is kept as a missing state (None) and rendered distinctly from a
    # measured zero — never fabricated as 0.
    buckets = []
    for c in range(ncols):
        s = c * n // ncols
        e = (c + 1) * n // ncols
        chunk = [v for v in window[s:e] if v is not None]
        buckets.append(sum(chunk) / len(chunk) if chunk else None)
    numeric = [b for b in buckets if b is not None]
    mx = max(numeric) if numeric else 1.0
    if mx == 0:
        mx = 1.0

    lines = [f"\n{label}",
             f"  (n={n} active, leading={lead}, trailing={trail}, max={mx:.3g}{gap_note})",
             "  legend: '#' >= threshold, ' ' measured below threshold, '.' no sample in bucket"]
    for r in range(max_rows, 0, -1):
        threshold = mx * r / max_rows
        row = "".join("." if b is None else ("#" if b >= threshold else " ") for b in buckets)
        lines.append(f"  {threshold:8.2g} | {row}")
    lines.append("  " + " " * 10 + "-" * len(buckets))
    lines.append("  " + " " * 10 + " (time -> ; indices are not timestamps)")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--metric", type=str, action="append", default=None,
                    help="Override: plot only these metric names.")
    ap.add_argument("--action", type=str, default=None,
                    help="Select one action (rXaY or index); default plots all.")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--cols", type=int, default=80)
    args = ap.parse_args(argv)

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    if len(set(args.tag)) != len(args.tag):
        ap.error("--tag values must be unique")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    out_lines = []
    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        capture = load_capture(rep)
        for action in select_actions(capture, args.action):
            label = action_label(action)
            pm_series = action.get("pm_series") or {}
            names = args.metric or sorted(pm_series)
            out_lines.append(f"\n{'=' * 60}\n{tag} {label}: {rep.name}\n{'=' * 60}")
            if not names:
                out_lines.append("  (no pmsampling:/warpsampling: series in this action)")
            for m in names:
                vals = instance_values(action, m)
                if vals is None:
                    out_lines.append(f"\n{m}: no instances")
                    continue
                out_lines.extend(ascii_plot(vals, m, args.rows, args.cols))

    out_path = analysis_dir / "pm_timeline_plots.txt"
    out_path.write_text("\n".join(out_lines))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
