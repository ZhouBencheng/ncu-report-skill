"""Shared helpers for parsing Nsight Compute reports (written against ncu 2026.1).

Usage:
    from ncu_utils import load_report, select_actions, safe, dump_all_metrics

`ncu_report` ships with Nsight Compute under `<install>/extras/python`. If it is
not importable already, it is found the way the CUDA toolkit's `ncu` launcher
finds Nsight Compute: from the `ncu` on PATH (the newest install it can reach).
`python3 ncu_utils.py` prints where it was found.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path


# --- Locate ncu_report -------------------------------------------------------

def _version(path: Path) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", path.name))


def ncu_python_dir() -> Path | None:
    """`<Nsight Compute install>/extras/python` for the `ncu` on PATH, or None."""
    ncu = shutil.which("ncu")
    if ncu is None:
        return None
    real = Path(ncu).resolve()
    if (real.parent / "extras/python/ncu_report.py").is_file():
        return real.parent / "extras/python"
    # The CUDA toolkit's `ncu` is a launcher script: it runs the newest Nsight Compute
    # of the .deb/.rpm installs (/opt/nvidia/nsight-compute/<version>) and of the
    # .run installs next to the toolkit (<toolkit>/nsight-compute-<version>).
    installs = [*Path("/opt/nvidia/nsight-compute").glob("*"),
                *real.parent.parent.glob("nsight-compute-*")]
    installs = [d for d in installs if (d / "extras/python/ncu_report.py").is_file()]
    if not installs:
        return None
    return max(installs, key=_version) / "extras/python"


try:
    import ncu_report  # noqa: F401
except ImportError:
    _found = ncu_python_dir()
    if _found is None:
        raise ImportError(
            "ncu_report is not importable and no Nsight Compute install was found from the "
            "`ncu` on PATH: put <Nsight Compute install>/extras/python on PYTHONPATH"
        ) from None
    sys.path.insert(0, str(_found))
    import ncu_report  # noqa: F401


# --- Loading -----------------------------------------------------------------

def actions(report):
    """Every action (= one profiled kernel launch) of a loaded report, in order."""
    out = []
    for r in range(report.num_ranges()):
        rng = report.range_by_idx(r)
        out.extend(rng.action_by_idx(a) for a in range(rng.num_actions()))
    return out


def load_report(path):
    """Load a .ncu-rep file and return (report, first action).

    Keep the report alive while using the action.
    """
    r = ncu_report.load_report(str(path))
    return r, actions(r)[0]


def load_action(path):
    """Shortcut when you don't need the report object separately."""
    _, action = load_report(path)
    return action


def select_actions(path, kernel=None):
    """(report, [actions]) of a report, keeping those whose function name matches `kernel`."""
    r = ncu_report.load_report(str(path))
    acts = actions(r)
    if kernel:
        pat = re.compile(kernel)
        acts = [a for a in acts if pat.search(kernel_name(a))]
    return r, acts


def kernel_name(action, form="demangled"):
    """The kernel's name: "demangled" (with template arguments), "function" or "mangled"."""
    base = {"demangled": action.NameBase_DEMANGLED, "function": action.NameBase_FUNCTION,
            "mangled": action.NameBase_MANGLED}[form]
    return action.name(base)


def action_tags(tag, acts):
    """One output tag per action: `tag` alone for one action, else `tag.<i>_<function>`."""
    if len(acts) == 1:
        return [tag]
    return [f"{tag}.{i}_{kernel_name(a, 'function')}" for i, a in enumerate(acts)]


# --- Safe metric access ------------------------------------------------------

def safe(action, name, default=None):
    """Return metric value, or `default` if the metric is missing or errors."""
    try:
        return action[name].value()
    except Exception:
        return default


def safe_many(action, names, default=None):
    """Bulk-fetch multiple metrics. Returns a dict name -> value-or-default."""
    return {n: safe(action, n, default) for n in names}


def metric_or_none(action, *candidates):
    """Try each candidate name, return first that works. Useful for
    GPU-gen-specific names: some metric names differ on sm_100 vs sm_90."""
    for n in candidates:
        v = safe(action, n, None)
        if v is not None:
            return v
    return None


# --- Value-kind robust accessor (for per-instance data) ---------------------

def metric_value_at(m, i):
    """Read the i-th instance value regardless of value kind."""
    k = m.kind()
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    if k == m.ValueKind_STRING:
        return m.as_string(i)
    # Fallbacks
    try:
        return m.as_uint64(i)
    except Exception:
        try:
            return m.as_double(i)
        except Exception:
            return None


def per_instance_values(action, metric_name):
    """Return a list of per-instance values, or None if the metric has none."""
    try:
        m = action[metric_name]
    except Exception:
        return None
    try:
        n = m.num_instances()
    except Exception:
        return None
    if n == 0:
        return None
    return [metric_value_at(m, i) for i in range(n)]


# --- Archive all metrics -----------------------------------------------------

def dump_all_metrics(action, outfile):
    """Dump every metric name + value to a JSON file for later analysis.

    Returns the number of entries written.
    """
    out = []
    for n in sorted(action.metric_names()):
        try:
            m = action[n]
            rec = {"name": n}
            try:
                rec["value"] = m.value()
            except Exception as e:
                rec["error"] = str(e)
            try:
                rec["unit"] = m.unit()
            except Exception:
                pass
            out.append(rec)
        except Exception as e:
            out.append({"name": n, "error": str(e)})
    Path(outfile).write_text(json.dumps(out, indent=1, default=str))
    return len(out)


# --- PC → source line mapping ------------------------------------------------

def per_pc_values(action, metric_name):
    """For a source-level metric (with correlation_ids = PCs), return list of (pc, value)."""
    try:
        m = action[metric_name]
    except Exception:
        return []
    try:
        n = m.num_instances()
    except Exception:
        return []
    if n == 0 or not m.has_correlation_ids():
        return []
    cor = m.correlation_ids()
    out = []
    for i in range(n):
        try:
            pc = cor.as_uint64(i)
        except Exception:
            try:
                pc = int(cor.as_double(i))
            except Exception:
                pc = None
        try:
            v = metric_value_at(m, i)
        except Exception:
            v = 0
        out.append((pc, v))
    return out


def pc_to_source_line(action, pc):
    """Return (file, line) for a given PC, or ('?', 0) if unavailable.

    Requires -lineinfo at compile time.
    """
    try:
        si = action.source_info(pc)
        if si is None:
            return "?", 0
        return si.file_name(), si.line()
    except Exception:
        return "?", 0


# --- Curated metric sets -----------------------------------------------------
#
# Every name below is collected by `ncu --set full` (ncu 2026.1) on Blackwell
# (sm_100 / sm_103). For a fuller list and rationale see
# ../reference/08-metric-names.md. Other GPU generations and future ncu releases
# may need alternate names: verify with action.metric_names() if one is None.

KEY_METRICS = [
    # Launch geometry
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
    # Timing and the clock the kernel ran at (ncu's --clock-control)
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg.per_second",
    "smsp__cycles_active.avg",
    # SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    # Occupancy
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.max.per_cycle_active",
    "sm__warps_active.min.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_eligible.max.per_cycle_active",
    # IPC
    "sm__inst_executed.avg.per_cycle_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.avg",
    # Compute pipes
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active",
    # Tensor core
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg",
    # DRAM
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",   # reads + writes
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    # Caches
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
    # Memory instruction counts
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    # Sectors / requests (for coalescing analysis)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
    # Stall reasons — aggregate ratios
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
    # Stall reasons — per-PC (SourceCounters section, part of --set full)
    "smsp__pcsamp_sample_count",
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_selected",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_membar",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
    "smsp__pcsamp_warps_issue_stalled_misc",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle",
]
B200_KEY_METRICS = KEY_METRICS  # the name older scripts import


# --- NCU rule results --------------------------------------------------------

_SEVERITY = {0: "INF", 1: "OK", 2: "OPT", 3: "WRN", 4: "ERR"}  # ncu_report.MsgType_*
_SPEEDUP_TYPE = {1: "local", 2: "global"}                      # ncu_report.SpeedupType_*


def _plain(message):
    """Rule text without ncu's "@section:<id>:<title>@" link markup."""
    return re.sub(r"@\w+:[^:@]*:([^@]*)@", r"\1", message or "")


def rule_speedups(action):
    """NCU rule-engine results as dicts, highest estimated speedup first.

    Keys: speedup (percent, None when the rule gives none), speedup_type ("local": of
    the part the rule is about, "global": of the whole kernel), severity
    (OPT/WRN/INF/OK/ERR), section, rule, title, message.
    """
    out = []
    for rr in action.rule_results():
        est = rr.speedup_estimation() if rr.has_speedup_estimation() else {}
        msg = rr.rule_message() if rr.has_rule_message() else {}
        if not msg.get("message") and not est:
            continue
        out.append({
            "speedup": est.get("speedup"),
            "speedup_type": _SPEEDUP_TYPE.get(est.get("type")),
            "severity": _SEVERITY.get(msg.get("type"), "?"),
            "section": rr.section_identifier(),
            "rule": rr.rule_identifier(),
            "title": msg.get("title", ""),
            "message": _plain(msg.get("message", "")),
        })
    out.sort(key=lambda r: -(r["speedup"] or 0.0))
    return out


# --- PM sampling time series -----------------------------------------------------

def pm_series_names(action):
    """Names of the PM-sampling time series in a report, sorted.

    ncu 2026.1 names them `pmsampling:<metric>` (PmSampling_WarpStates section) and
    `<UNIT>.<group>.<metric>` (PmSampling section, e.g.
    `TPC.TriageCompute.sm__inst_executed_realtime.avg.per_cycle_elapsed`).
    """
    pat = re.compile(r"^(pmsampling:|[A-Z][A-Z0-9_]*\.\w+\.)")
    out = []
    for n in action.metric_names():
        if pat.match(n):
            try:
                if action[n].num_instances() > 1:
                    out.append(n)
            except Exception:
                pass
    return sorted(out)


def pm_series(action, name):
    """(timestamps_ns, values) of one PM-sampling series, or None."""
    try:
        m = action[name]
        n = m.num_instances()
    except Exception:
        return None
    if n < 2:
        return None
    vals = [metric_value_at(m, i) for i in range(n)]
    ts = None
    if m.has_correlation_ids():
        cor = m.correlation_ids()
        ts = [cor.as_uint64(i) for i in range(n)]
    return ts, vals


# --- Inlined source lines (for CUTLASS/CuTe-style kernels) ---------------------

_FILE_RE = re.compile(r'//## File "([^"]+)", line (\d+)')
_INST_RE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;?\s*$")


def _opcode(sass):
    words = sass.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return words[0].split(".")[0] if words else ""


def inline_chains(action, binary):
    """Map each PC of the kernel to its inline chain, from the binary that was profiled.

    `action.source_info(pc)` gives only the innermost inlined frame (a library header
    line for CUTLASS/CuTe code); `nvdisasm -gi` gives every frame. `binary` is a cubin
    or anything `cuobjdump -xelf` can extract one from (executable, .so, fatbin).

    Returns (chains, checked, mismatched): chains maps pc -> [(file, line), ...] from
    the innermost frame to the outermost, which is the line in the kernel function
    itself; checked/mismatched count the sampled PCs whose SASS was compared with the
    report's. Needs cuobjdump and nvdisasm on PATH.
    """
    import subprocess
    import tempfile

    mangled = kernel_name(action, "mangled")
    # launch__function_pcs: one instance per function of the launch, the kernel first
    base = int(metric_value_at(action["launch__function_pcs"], 0))
    sampled = [pc for pc, _ in per_pc_values(action, "smsp__pcsamp_sample_count")]
    best = ({}, 0, 0)
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(binary).resolve()
        if binary.suffix == ".cubin":
            cubins = [binary]
        else:
            subprocess.run(["cuobjdump", "-xelf", "all", str(binary)], cwd=tmp,
                           check=True, capture_output=True)
            cubins = sorted(Path(tmp).glob("*.cubin"))
        for cubin in cubins:
            out = subprocess.run(["nvdisasm", "-gi", str(cubin)], capture_output=True,
                                 text=True).stdout
            insts, chain, in_func, block = {}, [], False, False
            for line in out.splitlines():
                if "\t.section\t.text." in line:
                    in_func = line.split(".text.", 1)[1].split(",")[0] == mangled
                    chain, block = [], False
                    continue
                if not in_func:
                    continue
                m = _FILE_RE.search(line)
                if m:
                    if not block:
                        chain = []
                    chain.append((m.group(1), int(m.group(2))))
                    block = True
                    continue
                block = False
                mi = _INST_RE.match(line)
                if mi:
                    insts[base + int(mi.group(1), 16)] = (mi.group(2), list(chain))
            if not insts:
                continue
            checked = [pc for pc in sampled if pc in insts]
            mismatched = sum(_opcode(insts[pc][0]) != _opcode(action.sass_by_pc(pc))
                             for pc in checked) + len(sampled) - len(checked)
            if not best[0] or mismatched < best[2]:
                best = ({pc: c for pc, (_, c) in insts.items()}, len(sampled), mismatched)
    return best


if __name__ == "__main__":
    print(f"ncu_report: {Path(ncu_report.__file__).parent}")
    print(f"ncu: {shutil.which('ncu')}")
