#!/usr/bin/env python3
"""Extract and compare key metrics and NCU's rule results from ncu reports (.ncu-rep / .ncu-repz).

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
import re
import sys
import textwrap
from pathlib import Path

# Make ncu_utils importable whether we're invoked from the skill dir or a run dir
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    action_tags, dump_all_metrics, kernel_name, rule_speedups,
    select_actions, select_target, metric_names, metric_evidence, ncu_report,
)


def collect(action, source: str, tag: str, analysis_dir: Path, target="auto", collector_version=None) -> dict:
    name = kernel_name(action)
    selection = select_target(action, target)
    print(f"[{tag}] {source}: {len(action.metric_names())} metrics, kernel {kernel_name(action, 'function')}")

    # Full archive
    n = dump_all_metrics(action, analysis_dir / f"metrics_all_{tag}.json")
    print(f"  -> metrics_all_{tag}.json ({n} metrics)")

    # Curated key metrics
    records = metric_evidence(action, metric_names(selection["target"]))
    key = {m: record["value"] for m, record in records.items()}
    key["__kernel_name__"] = name
    key["__target__"] = selection
    key["__metric_units__"] = {m: record["unit"] for m, record in records.items()}
    key["__metric_status__"] = {m: record["status"] for m, record in records.items()}
    reader_path = getattr(ncu_report, "__file__", None)
    bundled_version = re.search(r"nsight-compute[/-](\d+\.\d+\.\d+)", reader_path or "")
    module_version = getattr(ncu_report, "__version__", None)
    reader_version = module_version or (bundled_version.group(1) if bundled_version else None)
    reader_source = "module __version__" if module_version else "module path" if bundled_version else "unknown"
    evidence = {"target": selection,
                "collector": {"version": collector_version, "source": "collection-record argument" if collector_version else "unknown; not inferred from reader"},
                "reader": {"module": reader_path, "version": reader_version, "source": reader_source},
                "metrics": records}
    (analysis_dir / f"metrics_evidence_{tag}.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    print(f"  target={selection['target']} ({selection['status']}); unavailable fields retain explicit status")

    (analysis_dir / f"metrics_key_{tag}.json").write_text(
        json.dumps(key, indent=2, default=str)
    )
    with open(analysis_dir / f"metrics_key_{tag}.txt", "w") as f:
        f.write(f"===== {tag} =====\nKernel: {name}\n\n")
        for m, v in key.items():
            if m.startswith("__"):
                continue
            record = records[m]
            shown = str(v) if record["status"] == "present" else f"<{record['status']}>"
            unit = f" {record['unit']}" if record["unit"] else ""
            f.write(f"{m:95s} = {shown}{unit}\n")
    print(f"  -> metrics_key_{tag}.{{json,txt}}")

    # NCU's rule engine
    rules = rule_speedups(action)
    with open(analysis_dir / f"rules_{tag}.txt", "w") as f:
        f.write(f"===== NCU rules for {tag}, highest estimated speedup first =====\n")
        f.write("Kernel: " + name + "\n")
        f.write("speedup: est. % of the kernel's time (global) or of the part the rule is about (local)\n")
        if rules is None:
            f.write("\nUnavailable: this reader has no rule_results API; use the collector CLI details page.\n")
        for r in rules or []:
            sp = f"{r['speedup']:6.2f}% ({r['speedup_type']})" if r["speedup"] is not None else "     -"
            f.write(f"\n[{r['severity']}] {sp}  {r['section']} / {r['rule']}"
                    + (f": {r['title']}" if r["title"] else "") + "\n")
            f.write(textwrap.fill(r["message"], 110, initial_indent="    ",
                                  subsequent_indent="    ") + "\n")
    print(f"  -> rules_{tag}.txt ({str(len(rules)) + ' results' if rules is not None else 'reader API unavailable'})")
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
        f.write("Raw same-name metrics; compare units, normalization, workload and device before interpreting differences.\n")
        for m in dict.fromkeys(m for t in tags for m in collected[t] if not m.startswith("__")):
            statuses = {t: collected[t].get("__metric_status__", {}).get(
                m, "present" if collected[t].get(m) is not None else "not_selected") for t in tags}
            if all(statuses[t] != "present" for t in tags) and len(set(statuses.values())) == 1:
                continue  # collected for none of them (a survey report has only a few metrics)
            present_tags = [t for t in tags if statuses[t] == "present"]
            units = {t: collected[t].get("__metric_units__", {}).get(m)
                     for t in present_tags}
            known_units = {unit for unit in units.values() if unit is not None}
            if len(known_units) > 1:
                f.write(f"{m}: incompatible units {sorted(known_units)}; values not compared\n")
                continue
            missing_units = [t for t, unit in units.items() if unit is None]
            if len(present_tags) > 1 and missing_units:
                f.write(f"{m}: unit metadata unavailable for {', '.join(missing_units)}; "
                        "values not compared\n")
                continue
            f.write(f"{m:<95}")
            for t in tags:
                state = statuses[t]
                v = collected[t].get(m, "N/A") if state == "present" else f"<{state}>"
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
                    help="Regex on the kernel's function or demangled name: keep only matching kernels.")
    ap.add_argument("--target", default="auto",
                    help="Runtime SM/device alias; auto reads report identity, unknown uses generic. Conflicts fail.")
    ap.add_argument("--collector-version", action="append",
                    help="Recorded producer version, once per --report when supplied; never inferred from the reader")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    if len(set(args.tag)) != len(args.tag):
        ap.error("each --tag must be different: outputs are named after it")

    if args.collector_version and len(args.collector_version) != len(args.report):
        ap.error("when supplied, --collector-version must be passed once per --report")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    for rep, tag, collector_version in zip(args.report, args.tag, args.collector_version or [None] * len(args.report)):
        if not rep.exists():
            print(f"[skip] {rep} does not exist", file=sys.stderr)
            continue
        _report, acts = select_actions(rep, args.kernel)
        if not acts:
            print(f"[skip] {rep}: no kernel matches {args.kernel!r}", file=sys.stderr)
            continue
        for action, t in zip(acts, action_tags(tag, acts)):
            try:
                collected[t] = collect(action, rep.name, t, analysis_dir, args.target, collector_version)
            except ValueError as error:
                ap.error(str(error))

    compare(collected, analysis_dir)


if __name__ == "__main__":
    main()
