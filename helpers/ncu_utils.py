"""Shared helpers for parsing Nsight Compute reports (written against ncu 2026.3, reading 2026.1 reports too).

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
    """Load a report (.ncu-rep, or 2026.3's .ncu-repz) and return (report, first action).

    Keep the report alive while using the action.
    """
    r = ncu_report.load_report(str(path))
    return r, actions(r)[0]


def load_action(path):
    """Shortcut when you don't need the report object separately."""
    _, action = load_report(path)
    return action


def select_actions(path, kernel=None):
    """(report, [actions]) of a report, keeping those whose function or demangled name matches `kernel`."""
    r = ncu_report.load_report(str(path))
    acts = actions(r)
    if kernel:
        pat = re.compile(kernel)
        acts = [a for a in acts
                if pat.search(kernel_name(a, "function")) or pat.search(kernel_name(a))]
    return r, acts


def kernel_name(action, form="demangled"):
    """The kernel's name: "demangled" (with template arguments), "function" or "mangled"."""
    base = {"demangled": action.NameBase_DEMANGLED, "function": action.NameBase_FUNCTION,
            "mangled": action.NameBase_MANGLED}[form]
    return action.name(base)


def action_tags(tag, acts):
    """One output tag per action: `tag` alone for one action, else `tag.<i>_<function>`.

    The function name is cut to 60 characters of [A-Za-z0-9_.-], so file names stay short.
    """
    if len(acts) == 1:
        return [tag]
    return [f"{tag}.{i}_{re.sub(r'[^A-Za-z0-9_.-]', '_', kernel_name(a, 'function'))[:60]}"
            for i, a in enumerate(acts)]


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
            v = None
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
# Existing imports retain the B200/B300 list. New consumers select by report identity.
from ncu_profiles import metric_names, select_target, metric_evidence  # noqa: E402

KEY_METRICS = metric_names("sm100")
B200_KEY_METRICS = KEY_METRICS


# --- NCU rule results --------------------------------------------------------

_SEVERITY = {0: "INF", 1: "OK", 2: "OPT", 3: "WRN", 4: "ERR"}  # ncu_report.MsgType_*
_SPEEDUP_TYPE = {1: "local", 2: "global"}                      # ncu_report.SpeedupType_*


def _link(m):
    kind, _, rest = m.group(1, 2, 3)
    if kind == "url":  # @url:<text>:<address>@
        text, _, address = rest.partition(":")
        return f"{text} ({address})" if address else text
    return rest.rsplit(":", 1)[-1]  # @section:<id>:<title>@ and the like: the title


def _plain(message):
    """Rule text with ncu's "@<kind>:...@" link markup turned into plain text."""
    return re.sub(r"@(\w+)(:)([^@]*)@", _link, message or "")


def rule_speedups(action):
    """NCU rule-engine results as dicts, highest estimated speedup first.

    Keys: speedup (percent, None when the rule gives none), speedup_type ("local": of
    the part the rule is about, "global": of the whole kernel), severity
    (OPT/WRN/INF/OK/ERR), section, rule, title, message.
    """
    read_rules = getattr(action, "rule_results", None)
    if not callable(read_rules):
        return None
    try:
        results = read_rules()
    except Exception:
        return None
    out = []
    try:
        for rr in results:
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
    except Exception:
        return None
    out.sort(key=lambda r: -(r["speedup"] or 0.0))
    return out


# --- PM sampling time series -----------------------------------------------------

def pm_series_names(action):
    """Names of the PM-sampling time series in a report, sorted.

    The PmSampling section names them `<UNIT>.<group>.<metric>` (e.g.
    `TPC.TriageCompute.sm__inst_executed_realtime.avg.per_cycle_elapsed`); reports of ncu
    before 2026.3 may also have `pmsampling:smsp__warps_issue_stalled_*` (the
    PmSampling_WarpStates section, which 2026.3 removed).
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
    """(timestamps_ns, values) of one PM-sampling series, or None.

    With ncu_report 2026.3+ the series is aligned to the workload's timestamps
    (`action.timelines().aligned_metric`, context-switch filtered), else it is raw.
    """
    try:
        m = action[name]
    except Exception:
        return None
    read_timelines = getattr(action, "timelines", None)
    if callable(read_timelines):
        try:
            timelines = read_timelines()
            align = getattr(timelines, "aligned_metric", None)
            if callable(align):
                m = align(m) or m
        except Exception:
            pass  # Older readers can expose the method without supporting this report.
    try:
        n = m.num_instances()
    except Exception:
        return None
    if n < 2:
        return None
    vals = [metric_value_at(m, i) for i in range(n)]
    ts = None
    try:
        if m.has_correlation_ids():
            cor = m.correlation_ids()
            ts = [cor.as_uint64(i) for i in range(n)]
    except Exception:
        ts = None
    return ts, vals


def stall_reason_names():
    """ncu_report's StallReason codes -> names (long_scoreboard, wait, ...)."""
    return {getattr(ncu_report, k): k[len("StallReason_"):].lower()
            for k in dir(ncu_report) if k.startswith("StallReason_")}


def timed_warp_samples(action):
    """Timed warp samples, or None when the selected reader cannot expose them."""
    read_samples = getattr(action, "timed_warp_samples", None)
    if not callable(read_samples):
        return None
    try:
        return read_samples()
    except Exception:
        return None


def warp_state_timeline(action, buckets):
    """Stall reasons over the kernel's run, from its timed warp samples (PmSampling).

    Returns (span_ns, {reason: [share of the samples in each time bucket]}), or None
    when the report has no timed warp samples.
    """
    samples = timed_warp_samples(action)
    if not samples:
        return None
    t0 = min(s["timestamp"] for s in samples)
    span = max(s["timestamp"] for s in samples) - t0 or 1
    names = stall_reason_names()
    counts, totals = {}, [0] * buckets
    for s in samples:
        b = min(buckets - 1, (s["timestamp"] - t0) * buckets // span)
        totals[b] += 1
        reason = names.get(s["stall_reason"], str(s["stall_reason"]))
        counts.setdefault(reason, [0] * buckets)[b] += 1
    return span, {r: [c / t if t else None for c, t in zip(cs, totals)] for r, cs in counts.items()}


# --- Inlined source lines (for CUTLASS/CuTe-style kernels) ---------------------

_FILE_RE = re.compile(r'//## File "([^"]+)", line (\d+)')
_INST_RE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;?\s*$")


def _opcode(sass):
    words = sass.split()
    if words and words[0].startswith("@"):
        words = words[1:]
    return words[0].split(".")[0] if words else ""


class BinaryError(Exception):
    """The binary given for inline chains cannot be read (missing, or no device code)."""


_extracted = {}    # (binary, arch) -> cubin paths
_disassembly = {}  # cubin path -> nvdisasm -gi text


def _run(cmd, **kw):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise BinaryError(f"{' '.join(cmd[:2])} {cmd[-1]}: " + (r.stderr or r.stdout).strip())
    return r.stdout


def _cubins(binary, arch):
    """The cubins of `binary` for sm_<arch>: that architecture's if any, else its family's, else all."""
    import atexit
    import shutil as _shutil
    import tempfile

    key = (binary, arch)
    if key not in _extracted:
        if binary.suffix == ".cubin":
            _extracted[key] = [binary]
            return _extracted[key]
        names = re.findall(r"ELF file\s+\d+:\s+(\S+)", _run(["cuobjdump", "-lelf", str(binary)]))
        if not names:
            raise BinaryError(f"{binary} holds no cubin (cuobjdump -lelf lists none)")
        chosen = ([n for n in names if f".sm_{arch}" in n]
                  or [n for n in names if f".sm_{arch[:-1]}" in n] or names)
        tmp = tempfile.mkdtemp(prefix="ncu_utils_cubins_")
        atexit.register(_shutil.rmtree, tmp, True)
        _run(["cuobjdump", "-xelf", ",".join(chosen), str(binary)], cwd=tmp)
        _extracted[key] = sorted(Path(tmp).glob("*.cubin"))
    return _extracted[key]


def inline_chains(action, binary):
    """Map each PC of the kernel to its inline chain, from the binary that was profiled.

    `action.source_info(pc)` gives one frame, mostly the innermost inlined one (a library header
    line for CUTLASS/CuTe code); `nvdisasm -gi` gives every frame. `binary` is a cubin
    or anything `cuobjdump -xelf` can extract one from (executable, .so, fatbin); only the
    cubins of the report's GPU architecture that define the kernel are disassembled, once.

    Returns (chains, checked, mismatched): chains maps pc -> [(file, line), ...] from
    the innermost frame to the outermost, which is the line in the kernel function
    itself; checked/mismatched count the sampled PCs whose SASS was compared with the
    report's (a PC outside the kernel's own code, such as a non-inlined device call,
    counts as mismatched). Needs cuobjdump and nvdisasm on PATH; raises BinaryError
    when the binary cannot be read.
    """
    binary = Path(binary).resolve()
    if not binary.is_file():
        raise BinaryError(f"{binary} does not exist")
    mangled = kernel_name(action, "mangled")
    arch = (f"{int(action['device__attribute_compute_capability_major'].value())}"
            f"{int(action['device__attribute_compute_capability_minor'].value())}")
    # launch__function_pcs: one instance per function of the launch, the kernel first
    base = int(metric_value_at(action["launch__function_pcs"], 0))
    sampled = [pc for pc, _ in per_pc_values(action, "smsp__pcsamp_sample_count")]
    best = ({}, 0, 0)
    for cubin in _cubins(binary, arch):
        if mangled.encode() not in cubin.read_bytes():
            continue
        if cubin not in _disassembly:
            _disassembly[cubin] = _run(["nvdisasm", "-gi", str(cubin)])
        insts, chain, in_func, block = {}, [], False, False
        for line in _disassembly[cubin].splitlines():
            if line.startswith("\t.section\t"):  # any section ends the kernel's code
                in_func = (line.startswith("\t.section\t.text.")
                           and line.split(".text.", 1)[1].split(",")[0] == mangled)
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
