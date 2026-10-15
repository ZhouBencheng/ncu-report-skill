#!/usr/bin/env python3
"""Aggregate per-PC stall samples into per-source-line hotspots.

Consumes the detached capture's ``pc_analysis`` (produced once per report in a
fresh process). The capture already aggregates ordinary stall states per PC
without zero-fill or top-k truncation, keeps the not-issued (overlapping) subset
OUT of the totals, and records whether each PC mapped to a source line. This CLI
rolls those rows up by source line for EVERY range/action and reports true
coverage; PCs that did not map (e.g. a report compiled without ``-lineinfo``) are
retained and reported as source-mapping-unavailable rather than fabricated.

Usage:
    python3 extract_stall_hotspots.py --run-dir profile/myrun \\
            --report profile/myrun/reports/source_<tag>.ncu-rep --tag <tag>
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import action_name, action_slug, load_capture, select_actions  # noqa: E402


def short_stall_name(full):
    return full.replace("smsp__pcsamp_warps_issue_stalled_", "")


def rollup(action):
    """Return (per_line, unmapped_pcs, pca) from the capture's pc_analysis.

    per_line: dict[(file, line)] -> dict[stall -> count] for mapped PCs.
    unmapped_pcs: list of (pc, total, stalls) for PCs with no source line.
    """
    pca = action.get("pc_analysis") or {}
    per_line = defaultdict(lambda: defaultdict(int))
    unmapped = []
    for row in pca.get("rows") or []:
        stalls = row.get("ordinary_states") or {}
        if row.get("source_mapped") and isinstance(row.get("source"), dict):
            key = (row["source"].get("file"), row["source"].get("line"))
            for sn, v in stalls.items():
                per_line[key][sn] += v
        else:
            unmapped.append((row.get("pc"), row.get("ordinary_state_samples", 0), stalls))
    return per_line, unmapped, pca


def write_report(per_line, unmapped, pca, out_path, label, top_n=30):
    totals = [(sum(s.values()), f_, ln, dict(s)) for (f_, ln), s in per_line.items()]
    totals.sort(key=lambda x: -x[0])

    with open(out_path, "w") as f:
        f.write(f"===== Stall hotspots for {label} =====\n")
        f.write(f"Ordinary stall states exclude the not-issued subset: "
                f"{pca.get('overlapping_not_issued_names_excluded_from_totals') or []}\n")
        total_samples = pca.get("ordinary_state_samples")
        coverage = pca.get("sample_coverage")
        f.write(f"Total PCs: {pca.get('pc_count')}  "
                f"mapped PCs: {pca.get('mapped_pc_count')}  "
                f"ordinary-state samples: {total_samples}  "
                f"source coverage: {coverage}\n")
        f.write(f"Distinct mapped (file, line) entries: {len(totals)}\n\n")

        f.write(f"{'Rank':>4} {'Total':>10} {'File':<30} {'Line':>5}  Breakdown (stall: count)\n")
        f.write("-" * 150 + "\n")
        for i, (total, fn, ln, stalls) in enumerate(totals[:top_n]):
            short = Path(str(fn)).name if fn and fn != "?" else "?"
            breakdown = ", ".join(f"{short_stall_name(k)}: {v}"
                                  for k, v in sorted(stalls.items(), key=lambda x: -x[1]))
            f.write(f"{i:>4} {total:>10} {short:<30} {ln:>5}  {breakdown}\n")

        if unmapped:
            f.write(f"\n\n===== Source mapping UNAVAILABLE for {len(unmapped)} PC(s) "
                    f"(e.g. compiled without -lineinfo) =====\n")
            f.write("PCs are retained with samples; source lines are not fabricated.\n")
            for pc, tot, _ in sorted(unmapped, key=lambda x: -x[1])[:top_n]:
                pc_str = hex(pc) if isinstance(pc, int) else str(pc)
                f.write(f"  {tot:>10}  pc={pc_str}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--action", type=str, default=None,
                    help="Select one action (rXaY or index); default processes all.")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv)

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    if len(set(args.tag)) != len(args.tag):
        ap.error("--tag values must be unique; a repeated tag would overwrite an "
                 "earlier report's hotspot output")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        capture = load_capture(rep)
        for action in select_actions(capture, args.action):
            slug = action_slug(tag, action)
            per_line, unmapped, pca = rollup(action)
            out = analysis_dir / f"stall_hotspots_{slug}.txt"
            write_report(per_line, unmapped, pca, out, f"{slug} ({action_name(action)})",
                         top_n=args.top)
            print(f"[{slug}] -> {out} ({len(per_line)} mapped lines, {len(unmapped)} unmapped PCs)")


if __name__ == "__main__":
    main()
