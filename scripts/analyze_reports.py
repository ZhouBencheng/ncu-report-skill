#!/usr/bin/env python3
"""Extract and compare key metrics from .ncu-rep files.

Each report is read once by the detached capture child (fresh process; the only
native importer), then EVERY range/action is archived — no first-action-only
collapse. For each action produces, in ``<run-dir>/analysis/``:
    metrics_all_<tag>_<label>.json  — every metric with typed value, unit, IDs,
                                      per-instance values/kinds, correlation and
                                      per-read errors (archival; lossless)
    metrics_key_<tag>_<label>.{json,txt} — curated B200-compatible key metrics
    compare_<...>.txt (when >= 2 actions) — side-by-side

Pass --action rXaY (or a flat index) to archive a single action explicitly; an
ambiguous or non-existent selector fails loudly instead of defaulting to 0.

Usage:
    python3 analyze_reports.py --run-dir profile/myrun \\
            --report profile/myrun/reports/full_<tag>.ncu-rep --tag <tag>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    B200_KEY_METRICS, GENERIC_KEY_METRICS, PROFILE_CHOICES, PROFILE_METRICS,
    action_name, action_slug, load_capture, metric_display, metric_unit,
    metric_unit_display, metric_value, read_identity, select_actions,
    select_profile, unit_is_known,
)


def collect_action(action, tag, analysis_dir, key_metrics=B200_KEY_METRICS,
                   selection=None):
    name = action_name(action)
    slug = action_slug(tag, action)
    metrics = action.get("metrics") or {}
    print(f"[{slug}] {len(metrics)} metrics, kernel {name}")

    # Full lossless archive: the detached metric records already carry typed
    # value, unit, per-instance values/kinds, correlation IDs/units and errors.
    # This write is independent of the selected profile — the raw archive is
    # byte-content-equivalent regardless of --profile.
    (analysis_dir / f"metrics_all_{slug}.json").write_text(
        json.dumps(metrics, indent=1))
    print(f"  -> metrics_all_{slug}.json ({len(metrics)} metrics)")

    # Key JSON is a LOSSY scalar convenience projection: it keeps raw typed
    # scalar values only, and deliberately omits native units and scalar
    # availability. A bare number here does NOT establish that a measurement is
    # available (an unavailable/true-zero collapse is possible), and it carries
    # no unit — a consumer needing value/unit/availability/error semantics must
    # read the same-slug `metrics_all_*.json` typed archive instead. The TXT view
    # below is the availability-aware human curated view: absent / failed-read /
    # scalar-unavailable is never shown as 0, and each value carries its native
    # unit (with an explicit unknown/unreadable marker). The curated metric set
    # is the selected profile's list (default: B200).
    key = {m: metric_value(action, m) for m in key_metrics}
    key["__kernel_name__"] = name
    key["__projection__"] = ("lossy scalar convenience projection — omits native "
                             "units and scalar availability; see metrics_all_"
                             f"{slug}.json for value/unit/availability/error")
    if selection is not None:
        key["__profile_selection__"] = selection
    (analysis_dir / f"metrics_key_{slug}.json").write_text(json.dumps(key, indent=2))
    with open(analysis_dir / f"metrics_key_{slug}.txt", "w") as f:
        f.write(f"===== {slug} =====\nKernel: {name}\n")
        if selection is not None:
            f.write(f"Profile: {selection['chosen_profile']} "
                    f"({selection['status']}) — {selection['reason']}\n")
        f.write("\n")
        for m in key_metrics:
            line = f"{m:95s} = {metric_display(action, m)}"
            unit_disp = metric_unit_display(action, m)
            if unit_disp:
                line += f" {unit_disp}"
            f.write(line + "\n")
    print(f"  -> metrics_key_{slug}.{{json,txt}}")
    return slug


def compare(collected, analysis_dir, key_metrics=B200_KEY_METRICS):
    slugs = list(collected.keys())
    if len(slugs) < 2:
        return
    out_path = analysis_dir / f"compare_{'_vs_'.join(slugs)}.txt"
    with open(out_path, "w") as f:
        col_w = max(20, max(len(s) for s in slugs) + 2)
        f.write(f"{'Metric':<95}")
        for s in slugs:
            f.write(f"{s:>{col_w}}")
        f.write("\n" + "-" * (95 + col_w * len(slugs)) + "\n")
        for m in key_metrics:
            # Attach each numeric cell's native unit and suppress the comparison
            # when the compared actions do not share one known unit: mismatched,
            # absent or unreadable units must never be presented as directly
            # comparable bare numbers (no normalization, no inferred unit).
            cells, units_seen, numeric_count, unit_unknown = [], set(), 0, False
            for s in slugs:
                action = collected[s]
                disp = metric_display(action, m)
                numeric = isinstance(disp, (int, float)) and not isinstance(disp, bool)
                unit = metric_unit(action, m)
                # A unit is comparable only if it is a readable string. A missing
                # (None) or UNREADABLE (native-error dict, which is unhashable)
                # unit must not be added to the set — doing so would raise — and
                # marks the row non-comparable instead.
                known = unit_is_known(unit)
                if numeric:
                    numeric_count += 1
                    if known:
                        units_seen.add(unit)
                        label = unit if unit != "" else "unit: empty"
                    else:
                        unit_unknown = True
                        label = "unit: read-error" if isinstance(unit, dict) else "?"
                    text = f"{disp:.4g} [{label}]"
                else:
                    text = str(disp)
                cells.append(text)
            comparable = numeric_count >= 2 and not unit_unknown and len(units_seen) <= 1
            f.write(f"{m:<95}")
            for text in cells:
                f.write(f"{text:>{col_w}}")
            if numeric_count >= 2 and not comparable:
                f.write("  <units differ or unknown — not comparable>")
            f.write("\n")
    print(f"compare -> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract key NCU metrics and compare.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True)
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--action", type=str, default=None,
                    help="Select one action (rXaY or index); default archives all.")
    ap.add_argument("--profile", choices=PROFILE_CHOICES, default="auto",
                    help="Curated key-metric profile. 'auto' (default) selects a "
                         "profile only on a qualified exact device identity, else "
                         "'generic' (never B200 by default). An explicit profile "
                         "that conflicts with a readable identity exits nonzero.")
    args = ap.parse_args(argv)

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")
    if len(set(args.tag)) != len(args.tag):
        ap.error("--tag values must be unique; a repeated tag would overwrite an "
                 "earlier report's archive and drop it from the comparison")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    collected = {}
    chosen_profiles = set()
    exit_code = 0
    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} does not exist", file=sys.stderr)
            continue
        capture = load_capture(rep)
        for action in select_actions(capture, args.action):
            identity = read_identity(action)
            chosen, status, reason = select_profile(args.profile, identity)
            if status == "conflict":
                exit_code = 1
            chosen_profiles.add(chosen)
            selection = {
                "requested_profile": args.profile, "chosen_profile": chosen,
                "status": status, "reason": reason,
                "identity": {
                    "display_name": identity["display_name"], "cc": identity["cc"],
                    "display_state": identity["display_state"],
                    "cc_major_state": identity["cc_major_state"],
                    "cc_minor_state": identity["cc_minor_state"],
                },
            }
            slug = collect_action(action, tag, analysis_dir,
                                  PROFILE_METRICS[chosen], selection)
            (analysis_dir / f"profile_selection_{slug}.json").write_text(
                json.dumps(selection, indent=2))
            print(f"  -> profile_selection_{slug}.json [{status} -> {chosen}]")
            collected[slug] = action

    # Cross-action compare table uses the profile actually CHOSEN per action,
    # NOT the requested one: a conflict downgrades every action to generic, and
    # the compare view must agree with those per-action views/sidecars rather
    # than render the refused profile's metrics. Homogeneous chosen profile ->
    # that list; mixed (or none) -> generic.
    if len(chosen_profiles) == 1:
        compare_metrics = PROFILE_METRICS[next(iter(chosen_profiles))]
    else:
        compare_metrics = GENERIC_KEY_METRICS
    compare(collected, analysis_dir, compare_metrics)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
