#!/usr/bin/env python3
"""Aggregate per-PC stall samples into per-source-line hotspots.

Requires:
    - a .ncu-rep with the SourceCounters section (`--set full` has it, or
      `--section SourceCounters` alone)
    - kernel compiled with `-lineinfo` (or `--generate-line-info`)

Produces in `<run-dir>/analysis/`, per profiled kernel:
    stall_hotspots_<tag>.txt — top lines ranked by total stall samples,
                               with per-stall-type breakdown, plus per-stall
                               top lines.

ncu maps a PC only to its innermost inlined frame, so in CUTLASS/CuTe-style code
most samples land on library header lines. `--binary` (the executable, .so or
cubin that was profiled) adds a ranking by the line in the kernel function itself,
from `nvdisasm -gi`.

A report that holds several kernels gives one file per kernel, tagged
`<tag>.<i>_<function>`; `--kernel REGEX` keeps the matching ones.

Usage:
    python3 extract_stall_hotspots.py --run-dir "$PROFILE_RUN_DIR" \\
            --report "$PROFILE_RUN_DIR/reports/full_<tag>.ncu-rep" --tag <tag> \\
            [--binary path/to/the/profiled/library.so]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ncu_utils import (  # noqa: E402
    action_tags, inline_chains, kernel_name, metric_value_at, select_actions,
)


STALL_METRICS = [
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
    "smsp__pcsamp_warps_issue_stalled_misc",
    "smsp__pcsamp_warps_issue_stalled_membar",
]


def collect_per_pc(action):
    """Return dict[pc] -> dict[stall_name -> count]."""
    per_pc = defaultdict(lambda: defaultdict(int))
    for sn in STALL_METRICS:
        try:
            m = action[sn]
        except Exception:
            continue
        try:
            n = m.num_instances()
        except Exception:
            continue
        if n == 0 or not m.has_correlation_ids():
            continue
        cor = m.correlation_ids()
        for i in range(n):
            try:
                pc = cor.as_uint64(i)
            except Exception:
                try:
                    pc = int(cor.as_double(i))
                except Exception:
                    continue
            v = metric_value_at(m, i)
            if v:
                per_pc[pc][sn] += int(v)
    return per_pc


def aggregate_by_source_line(action, per_pc):
    per_line = defaultdict(lambda: defaultdict(int))
    for pc, stalls in per_pc.items():
        try:
            si = action.source_info(pc)
            if si is None:
                file_, line = "?", 0
            else:
                file_, line = si.file_name(), si.line()
        except Exception:
            file_, line = "?", 0
        for sn, v in stalls.items():
            per_line[(file_, line)][sn] += v
    return per_line


def aggregate_by_kernel_line(per_pc, chains):
    """Per (file, line) of the outermost inline frame: the kernel function's own line."""
    per_line = defaultdict(lambda: defaultdict(int))
    for pc, stalls in per_pc.items():
        chain = chains.get(pc)
        key = chain[-1] if chain else ("?", 0)
        for sn, v in stalls.items():
            per_line[key][sn] += v
    return per_line


def short_stall_name(full):
    return full.replace("smsp__pcsamp_warps_issue_stalled_", "")


def _write_ranking(f, per_line, top_n):
    totals = []
    for (f_, ln), stalls in per_line.items():
        total = sum(stalls.values())
        totals.append((total, f_, ln, dict(stalls)))
    totals.sort(key=lambda x: -x[0])

    f.write(f"Total distinct (file, line) entries: {len(totals)}\n\n")
    f.write(f"{'Rank':>4} {'Total':>10} {'File':<30} {'Line':>5}  Breakdown (stall: count)\n")
    f.write("-" * 150 + "\n")
    for i, (total, fn, ln, stalls) in enumerate(totals[:top_n]):
        short = Path(str(fn)).name if fn and fn != "?" else "?"
        breakdown = ", ".join(
            f"{short_stall_name(k)}: {v}"
            for k, v in sorted(stalls.items(), key=lambda x: -x[1]) if v
        )
        f.write(f"{i:>4} {total:>10} {short:<30} {ln:>5}  {breakdown}\n")

    f.write("\n\n===== Per-stall-type top lines =====\n")
    for sn in STALL_METRICS:
        items = [((fn, ln), s.get(sn, 0)) for (fn, ln), s in per_line.items()]
        items = [it for it in items if it[1] > 0]
        items.sort(key=lambda x: -x[1])
        if not items:
            continue
        f.write(f"\n--- {short_stall_name(sn)} ---\n")
        for (fn, ln), v in items[:10]:
            short = Path(str(fn)).name if fn and fn != "?" else "?"
            f.write(f"  {v:>8}  {short}:{ln}\n")


def write_report(per_line, out_path, tag, kernel, top_n=30, kernel_lines=None, alignment=None):
    with open(out_path, "w") as f:
        f.write(f"===== Stall hotspots for {tag} =====\nKernel: {kernel}\n")
        if kernel_lines is not None:
            checked, mismatched = alignment
            f.write(f"\n===== By line in the kernel function (outermost inline frame, from --binary; "
                    f"SASS of {checked - mismatched}/{checked} sampled PCs matched the report) =====\n")
            _write_ranking(f, kernel_lines, top_n)
            f.write("\n\n===== By innermost source line (ncu's own attribution) =====\n")
        _write_ranking(f, per_line, top_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, action="append", required=True,
                    help="Path(s) to source-level .ncu-rep file(s). Pass multiple with repeated flag.")
    ap.add_argument("--tag", type=str, action="append", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--kernel", type=str, default=None,
                    help="Regex on the demangled kernel name: keep only matching kernels.")
    ap.add_argument("--binary", type=Path, default=None,
                    help="The executable, .so or cubin that was profiled: also rank by the "
                         "kernel function's own lines (needs cuobjdump and nvdisasm).")
    args = ap.parse_args()

    if len(args.report) != len(args.tag):
        ap.error("--report and --tag counts must match")

    analysis_dir = args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for rep, tag in zip(args.report, args.tag):
        if not rep.exists():
            print(f"[skip] {rep} not found", file=sys.stderr)
            continue
        _report, acts = select_actions(rep, args.kernel)
        for action, t in zip(acts, action_tags(tag, acts)):
            per_pc = collect_per_pc(action)
            if not per_pc:
                print(f"[{t}] no per-PC stall samples: collect with --set full or "
                      f"--section SourceCounters", file=sys.stderr)
                continue
            per_line = aggregate_by_source_line(action, per_pc)
            if set(per_line) == {("?", 0)}:
                print(f"[{t}] no source lines: the kernel has no line info (built without "
                      f"-lineinfo); see 02-harness-guide.md", file=sys.stderr)
            kernel_lines = alignment = None
            if args.binary:
                chains, checked, mismatched = inline_chains(action, args.binary)
                if not chains:
                    print(f"[{t}] {args.binary} holds no SASS for this kernel", file=sys.stderr)
                else:
                    if mismatched:
                        print(f"[{t}] warning: SASS of {mismatched}/{checked} sampled PCs differs "
                              f"from the report: is {args.binary} the binary that was profiled?",
                              file=sys.stderr)
                    kernel_lines = aggregate_by_kernel_line(per_pc, chains)
                    alignment = (checked, mismatched)
            out = analysis_dir / f"stall_hotspots_{t}.txt"
            write_report(per_line, out, t, kernel_name(action), args.top, kernel_lines, alignment)
            print(f"[{t}] -> {out} ({len(per_line)} distinct lines)")


if __name__ == "__main__":
    main()
