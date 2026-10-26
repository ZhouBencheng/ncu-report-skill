"""Target routing and candidate metrics; no NCU reader or GPU is needed to import this module."""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent.parent / "reference"
IDENTITY_METRICS = (
    "device__attribute_display_name",
    "device__attribute_compute_capability_major",
    "device__attribute_compute_capability_minor",
)


@lru_cache(maxsize=1)
def profiles():
    return {p.parent.name: json.loads(p.read_text())
            for p in sorted((REFERENCE / "architectures").glob("*/profile.json"))}


def canonical_target(value):
    value = re.sub(r"[\s_-]+", "", str(value).lower())
    if value in ("auto", "generic") or value in profiles():
        return value
    for target, profile in profiles().items():
        if value in [re.sub(r"[\s_-]+", "", alias.lower()) for alias in profile["aliases"]]:
            return target
    raise ValueError(f"Unknown runtime target {value!r}; use an exact SM or device alias. "
                     "Compiler a/f suffixes and broad families are not device identities.")


def metric_names(target):
    """Candidates, not a claim that a collector supports or collected every name."""
    target = canonical_target(target)
    common = json.loads((REFERENCE / "common-metrics.json").read_text())
    profile = profiles().get(target, {})
    metric_profile = profiles().get(profile.get("metric_profile"), profile)
    extras = metric_profile.get("extra_metrics", [])
    # Keep the historical B200/B300 order: architecture-specific Tensor metrics
    # follow the shared Tensor metrics. Other targets contain only shared names.
    anchor = "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg"
    i = common.index(anchor) + 1
    additional = json.loads((REFERENCE / profile["additional_metrics"]).read_text()) if profile.get("additional_metrics") else []
    return list(dict.fromkeys(common[:i] + extras + common[i:] + additional))


def metric_evidence(action, names):
    """Preserve missing, scalar-unavailable, read-error and true zero separately.

    A missing report field cannot distinguish unsupported from not collected.
    Raw names and units stay attached; ratios and percentages are never aliased.
    """
    present = set(action.metric_names())
    out = {}
    for name in names:
        record = {"status": "not_in_report", "value": None, "unit": None}
        if name in present:
            try:
                metric = action[name]
            except Exception as error:
                record.update(status="read_error", error=f"{type(error).__name__}: {error}")
            else:
                try:
                    record["unit"] = metric.unit()
                except Exception:
                    pass
                try:
                    if hasattr(metric, "has_value") and not metric.has_value():
                        record["status"] = "scalar_unavailable"
                    else:
                        value = metric.value()
                        if value is None or (isinstance(value, float) and not math.isfinite(value)):
                            record["status"] = "scalar_unavailable"
                        else:
                            record.update(status="present", value=value)
                except Exception as error:
                    record.update(status="read_error", error=f"{type(error).__name__}: {error}")
        out[name] = record
    return out


def _cc_component(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 and int(value) == value:
        return int(value)
    return None


def read_identity(action):
    records = metric_evidence(action, IDENTITY_METRICS)
    values = [records[n]["value"] for n in IDENTITY_METRICS]
    return {"name": values[0] if isinstance(values[0], str) else None,
            "cc_major": _cc_component(values[1]), "cc_minor": _cc_component(values[2])}


def route_identity(identity, requested="auto"):
    requested = canonical_target(requested)
    major, minor = identity.get("cc_major"), identity.get("cc_minor")
    name = (identity.get("name") or "").lower()
    named = {target for target, profile in profiles().items()
             if any(re.search(r"(?<![a-z0-9])" + re.escape(pattern) + r"(?![a-z0-9])", name)
                    for pattern in profile["name_patterns"])}
    cc = f"{major}.{minor}" if major is not None and minor is not None else None
    by_cc = next((target for target, p in profiles().items() if p["compute_capability"] == cc), None)
    if len(named) > 1:
        raise ValueError("Conflicting device names in report identity")
    by_name = next(iter(named), None)
    constraints = [t for t in (by_name, requested if requested not in ("auto", "generic") else None) if t]
    for target in constraints:
        expected = tuple(map(int, profiles()[target]["compute_capability"].split(".")))
        if any(actual is not None and actual != want for actual, want in zip((major, minor), expected)):
            raise ValueError(f"Target {target} conflicts with observed device identity {identity}")
    if requested not in ("auto", "generic") and by_name and requested != by_name:
        raise ValueError(f"Requested {requested} conflicts with device name identifying {by_name}")
    detected = by_cc or by_name
    target = (detected or "generic") if requested == "auto" else requested
    status = "detected" if detected and target == detected else "explicit-unverified" if target != "generic" else "generic"
    return {"target": target, "status": status, "device": identity,
            "runtime_compute_capability": cc,
            "reference": f"reference/architectures/{target}/guide.md" if target != "generic" else "reference/01-workflow.md"}


def select_target(action, requested="auto"):
    return route_identity(read_identity(action), requested)
