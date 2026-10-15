"""A fake, CPU-only stand-in for the native ``ncu_report`` module.

It implements exactly the surface that ``ncu_capture`` touches, so the capture
seam can be exercised on a CPU host with no NCU SDK. Specs are plain dicts, so a
test can craft the five distinguishable value states (absent / NaN / +inf /
true-zero / read-error), correlation-length mismatches, the not-issued subset,
and no-lineinfo reports.

Crucially, the fake metric RECORDS every ``has_value`` call that carries a
positional argument into ``module.has_value_arg_calls``; a correct capture never
calls ``has_value(index)`` (that overload does not exist on the real SDK), so
tests assert that list stays empty.
"""
from __future__ import annotations

import types


class Boom:
    """Marker: accessing a field holding this raises, modelling a native read error."""
    def __init__(self, message="native read error"):
        self.message = message


class FakeValueKind:
    """Distinct type so has_value(ValueKind) is distinguishable from has_value(index)."""
    def __init__(self, name):
        self.name = name


def _resolve(value):
    if isinstance(value, Boom):
        raise RuntimeError(value.message)
    return value


class FakeMetric:
    def __init__(self, module, spec):
        self._m = module
        self._s = spec

    def _inst(self, i):
        return self._s.get("instances", [])[i]

    def value(self, *args):
        if args:
            return _resolve(self._inst(args[0]).get("value"))
        return _resolve(self._s.get("value"))

    def kind(self, *args):
        if args:
            return _resolve(self._inst(args[0]).get("kind"))
        return _resolve(self._s.get("kind", "DOUBLE"))

    def has_value(self, *args):
        # Record any positional-argument call; the capture must never pass an
        # instance index here. has_value(ValueKind) is allowed by the real API.
        if args:
            self._m.has_value_arg_calls.append(
                {"is_value_kind": isinstance(args[0], FakeValueKind), "arg": repr(args[0])})
        return _resolve(self._s.get("has_value", True))

    def unit(self):
        return _resolve(self._s.get("unit"))

    def num_instances(self):
        return _resolve(self._s.get("num_instances", len(self._s.get("instances", []))))

    def name(self):
        return self._s.get("native_name", self._s.get("name"))

    def description(self):
        return self._s.get("description", "")

    def rollup_operation(self):
        return self._s.get("rollup_operation", "avg")

    def metric_type(self):
        return self._s.get("metric_type", "counter")

    def metric_subtype(self):
        return self._s.get("metric_subtype")

    def has_correlation_ids(self):
        return bool(self._s.get("correlation"))

    def correlation_ids(self):
        cor = self._s.get("correlation")
        if cor is None:
            return None
        if isinstance(cor, Boom):
            raise RuntimeError(cor.message)
        # A correlation is just another metric whose per-instance values are PCs.
        return FakeMetric(self._m, cor)


class FakeSource:
    def __init__(self, file_name, line):
        self._f, self._l = file_name, line

    def file_name(self):
        return _resolve(self._f)

    def line(self):
        return _resolve(self._l)


class FakeAction:
    NameBase_DEMANGLED = "DEMANGLED"
    NameBase_FUNCTION = "FUNCTION"

    def __init__(self, module, spec):
        self._m = module
        self._s = spec

    def metric_names(self):
        return list(self._s.get("metrics", {}).keys())

    def metric_by_name(self, name):
        spec = self._s.get("metrics", {}).get(name)
        if spec is None:
            return None  # absent metric
        if isinstance(spec, Boom):
            raise RuntimeError(spec.message)
        return FakeMetric(self._m, spec)

    def name(self, base):
        if base == self.NameBase_FUNCTION:
            return self._s.get("function_name", self._s.get("name", "kernel"))
        return self._s.get("name", "kernel")

    def source_info(self, pc):
        src = self._s.get("sources", {}).get(pc)
        if src is None:
            return None
        return FakeSource(src[0], src[1])

    def sass_by_pc(self, pc):
        return self._s.get("sass", {}).get(pc)

    def rule_results_as_dicts(self):
        return list(self._s.get("rule_results", []))


class FakeRange:
    def __init__(self, module, actions):
        self._m = module
        self._actions = actions

    def num_actions(self):
        return len(self._actions)

    def action_by_idx(self, j):
        return FakeAction(self._m, self._actions[j])


class FakeReport:
    def __init__(self, module, ranges):
        self._m = module
        self._ranges = ranges

    def num_ranges(self):
        return len(self._ranges)

    def range_by_idx(self, i):
        return FakeRange(self._m, self._ranges[i])


def make_module(ranges):
    """Build a fake ncu_report module exposing load_report(path) -> FakeReport.

    ``ranges`` is a list of ranges; each range is a list of action specs.
    """
    module = types.ModuleType("ncu_report")
    module.__file__ = "<fake_ncu>"
    module.has_value_arg_calls = []
    module.ValueKind_DOUBLE = FakeValueKind("DOUBLE")
    module.load_report = lambda path: FakeReport(module, ranges)
    return module


def capture_of(ranges):
    """Capture a fake report in-process, returning (capture_dict, fake_module)."""
    import ncu_capture
    module = make_module(ranges)
    cap = ncu_capture.capture("nonexistent-report.ncu-rep", ncu_report_module=module)
    return cap, module


# --- Convenient spec builders ------------------------------------------------

def scalar_metric(value, unit=None, **extra):
    spec = {"value": value, "unit": unit, "num_instances": 0, "instances": []}
    spec.update(extra)
    return spec


def pc_metric(name, pairs, unit="sample", correlation_unit="sample"):
    """A per-PC stall metric: pairs is a list of (pc, count)."""
    return {
        "native_name": name, "value": sum(c for _, c in pairs if isinstance(c, (int, float))),
        "unit": unit, "num_instances": len(pairs),
        "instances": [{"kind": "UINT64", "value": c} for _, c in pairs],
        "correlation": {
            "unit": correlation_unit, "num_instances": len(pairs),
            "instances": [{"kind": "UINT64", "value": pc} for pc, _ in pairs],
        },
    }
