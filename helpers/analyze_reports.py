#!/usr/bin/env python3
"""Extract and compare key metrics and NCU's rule results from .ncu-rep files.

Produces in `<run_dir>/analysis/`, per profiled kernel:
    metrics_all_<tag>.json     — every metric, archival
    metrics_key_<tag>.txt/json — curated key metrics
    rules_<tag>.txt            — NCU rule-engine results, highest estimated speedup first
    compare_<tag1>_vs_<tag2>.txt (when >= 2 kernels) — side-by-side, metrics some kernel has
                               (compare_<tag1>_and_<n>_more.txt for many kernels)

A report that holds several kernels (no `-k`/`-c 1` at collection) gives one set per
kernel, tagged `<tag>.<i>_<function>`; `--kernel REGEX` keeps the matching ones.

Usage examples:
    # Single report
    python3 analyze_reports.py --run-dir "$PROFILE_RUN_DIR" \\
            --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag>

    # Multiple reports → side-by-side compare
    python3 analyze_reports.py --run-dir "$PROFILE_RUN_DIR" \\
            --report "$PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep" --tag <tag1> \\
            --report "$PROFILE_RUN_DIR/reports/full_<tag2>.ncu-rep" --tag <tag2>
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# Make ncu_utils importable whether we're invoked from the skill dir or a run dir
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    KEY_METRICS, action_tags, dump_all_metrics, kernel_name, rule_speedups, safe,
    select_actions,
)


def collect(action, source: str, tag: str, analysis_dir: Path) -> dict:
    name = kernel_name(action)
    print(f"[{tag}] {source}: {len(action.metric_names())} metrics, kernel {kernel_name(action, 'function')}")

    # Full archive
    n = dump_all_metrics(action, analysis_dir / f"metrics_all_{tag}.json")
    print(f"  -> metrics_all_{tag}.json ({n} metrics)")

    # Curated key metrics
    key = {m: safe(action, m) for m in KEY_METRICS}
    key["__kernel_name__"] = name

    (analysis_dir / f"metrics_key_{tag}.json").write_text(
        json.dumps(key, indent=2, default=str)
    )
    with open(analysis_dir / f"metrics_key_{tag}.txt", "w") as f:
        f.write(f"===== {tag} =====\nKernel: {name}\n\n")
        for m, v in key.items():
            if m.startswith("__"):
                continue
            f.write(f"{m:95s} = {v}\n")
    print(f"  -> metrics_key_{tag}.{{json,txt}}")

    # NCU's rule engine
    rules = rule_speedups(action)
    with open(analysis_dir / f"rules_{tag}.txt", "w") as f:
        f.write(f"===== NCU rules for {tag}, highest estimated speedup first =====\n")
        f.write("Kernel: " + name + "\n")
        f.write("speedup: est. % of the kernel's time (global) or of the part the rule is about (local)\n")
        for r in rules:
            sp = f"{r['speedup']:6.2f}% ({r['speedup_type']})" if r["speedup"] is not None else "     -"
            f.write(f"\n[{r['severity']}] {sp}  {r['section']} / {r['rule']}"
                    + (f": {r['title']}" if r["title"] else "") + "\n")
            f.write(textwrap.fill(r["message"], 110, initial_indent="    ",
                                  subsequent_indent="    ") + "\n")
    print(f"  -> rules_{tag}.txt ({len(rules)} results)")
    return key


def compare(collected: dict, analysis_dir: Path):
    tags = list(collected.keys())
    if len(tags) < 2:
        return
    name = "_vs_".join(tags)
    if len(name) > 150:
        name = f"{tags[0]}_and_{len(tags) - 1}_more"
    out_path = analysis_dir / f"compare_{name}.txt"
    with open(out_path, "w") as f:
        col_w = max(20, max(len(t) for t in tags) + 2)
        f.write(f"{'Metric':<95}")
        for t in tags:
            f.write(f"{t:>{col_w}}")
        f.write("\n" + "-" * (95 + col_w * len(tags)) + "\n")
        for m in KEY_METRICS:
            if all(collected[t].get(m) is None for t in tags):
                continue  # collected for none of them (a survey report has only a few metrics)
            f.write(f"{m:<95}")
            for t in tags:
                v = collected[t].get(m, "N/A")
                if isinstance(v, (int, float)):
                    v = f"{v:.4g}"
                f.write(f"{str(v):>{col_w}}")
            f.write("\n")
    print(f"compare -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Extract key NCU metrics and rule results, and compare.")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="The profile run directory — outputs go to <run-dir>/analysis/")
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="Path to a .ncu-rep file. Can be passed multiple times.")
    ap.add_argument("--tag", type=str, action="append", required=True,
                    help="Short tag for each report. Must be passed once per --report.")
    ap.add_argument("--kernel", type=str, default=None,
                    help="Regex on the demangled kernel name: keep only matching kernels.")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} does not exist", file=sys.stderr)
            continue
        _report, acts = select_actions(rep, args.kernel)
        if not acts:
            print(f"[skip] {rep}: no kernel matches {args.kernel!r}", file=sys.stderr)
            continue
        for action, t in zip(acts, action_tags(tag, acts)):
            collected[t] = collect(action, rep.name, t, analysis_dir)

    compare(collected, analysis_dir)


if __name__ == "__main__":
    main()
