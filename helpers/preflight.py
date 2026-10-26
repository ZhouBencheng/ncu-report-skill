#!/usr/bin/env python3
"""Read-only profiling environment discovery. Does not launch kernels or alter clocks."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

from ncu_profiles import metric_names, route_identity


def run(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}


def collection_options(target, sets, sections, help_text, l4t_observed=False):
    """A proposed recipe from installed options; collection can still fail (e.g. permission)."""
    options = []
    if re.search(r"\bfull\b", sets):
        options += ["--set", "full"]
    else:
        for section in ("SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis", "SourceCounters"):
            if re.search(r"\b" + section + r"\b", sections):
                options += ["--section", section]
    if "--import-source" in help_text:
        options += ["--import-source", "yes"]
    if "--clock-control" in help_text:
        # Jetson clocks are platform controlled. L4T remains sufficient platform
        # evidence when nvidia-smi cannot identify the device; it does not identify
        # a particular Jetson model or runtime SM.
        if target in ("sm87", "sm110") or l4t_observed:
            options += ["--clock-control", "none"]
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu", default="ncu", help="Selected collector executable, not the report reader")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target", default="auto", help="Optional runtime target, checked against observed identity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable = shutil.which(args.ncu)
    if not executable:
        parser.error(f"Collector {args.ncu!r} not found; select its actual installed path")
    device_query = run(["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader"])
    identity = {"name": None, "cc_major": None, "cc_minor": None}
    for row in csv.reader(device_query["stdout"].splitlines()):
        if len(row) == 3 and row[0].strip() == str(args.device):
            identity["name"] = row[1].strip()
            if re.fullmatch(r"\s*\d+\.\d+\s*", row[2]):
                identity["cc_major"], identity["cc_minor"] = map(int, row[2].strip().split("."))
    try:
        selection = route_identity(identity, args.target)
    except ValueError as error:
        parser.error(str(error))
    queries = {key: run([executable, *flags]) for key, flags in (
        ("version", ["--version"]), ("sets", ["--list-sets"]),
        ("sections", ["--list-sections"]), ("help", ["--help"]),
        ("metrics", ["--devices", str(args.device), "--query-metrics", "--query-metrics-mode", "all"]))}
    metric_query = queries["metrics"]
    available = sorted(set(re.findall(r"^([a-z][a-z0-9]*__[A-Za-z0-9_.:]+)\s", metric_query["stdout"], re.MULTILINE)))
    metric_query["stdout"] = "Metric names retained in available_metrics; descriptions omitted."
    candidates = metric_names(selection["target"])
    discovery_ok = metric_query["returncode"] == 0 and bool(available)
    l4t = Path("/etc/nv_tegra_release")
    data = {
        "target": selection, "machine": platform.machine(), "platform": platform.platform(),
        "l4t": l4t.read_text().strip() if l4t.is_file() else None,
        "collector": str(Path(executable).resolve()),
        "queries": queries, "device_query": device_query,
        "available_metrics": available,
        "candidate_availability": {n: ("listed_by_query" if n in available else "not_listed") if discovery_ok else "unknown" for n in candidates},
        "collection_options": collection_options(selection["target"], queries["sets"]["stdout"],
                                                 queries["sections"]["stdout"], queries["help"]["stdout"],
                                                 l4t_observed=l4t.is_file()),
        "notes": ["Discovery is not proof of counter permission or successful collection.",
                  "Record power mode and actual clocks separately; no clock or power settings were changed.",
                  "A reader may be a different version from this collector. Record both."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"target": selection, "collector_version": queries["version"]["stdout"].strip(),
                      "metric_count": len(available), "collection_options": data["collection_options"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
