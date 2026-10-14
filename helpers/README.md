# Helpers

Reusable code for profiling harnesses and report analysis. See `../SKILL.md` for context. `HELPERS` below is this directory.

## C++ / CUDA

| File | Purpose |
|---|---|
| `harness_template.cu` | Starting point for a profiling harness. Copy into your run dir, fill in the `TODO(you)` sections. |
| `safetensors_loader.h` | Header-only safetensors reader (no external deps). Use from your harness to load real workload tensors. |

### Typical harness setup

```bash
cd "$PROFILE_RUN_DIR/harness"
cp "$HELPERS/harness_template.cu" my_kernel_harness.cu
cp "$HELPERS/safetensors_loader.h" .
# edit my_kernel_harness.cu to include your kernel + fill in main()
nvcc -O2 -std=c++17 -lineinfo -arch=sm_100a \
     my_kernel_harness.cu -o my_kernel_harness     # sm_100a: B200, sm_103a: B300
```

## Python

| File | Purpose |
|---|---|
| `ncu_utils.py` | Shared helpers: finds `ncu_report`, `load_report` / `select_actions` (every kernel of a report), `kernel_name`, `safe`, `per_pc_values`, `KEY_METRICS`, `rule_speedups`, `pm_series_names`, `inline_chains`, ... `python3 ncu_utils.py` prints where `ncu_report` was found |
| `analyze_reports.py` | Key metrics, NCU rule results (by estimated speedup) and side-by-side comparison from one or more `.ncu-rep`s |
| `extract_stall_hotspots.py` | Aggregate per-PC stall samples → per-source-line rankings; `--binary` ranks the kernel function's own lines for inlined (CUTLASS/CuTe) code |
| `plot_timeline.py` | ASCII plot PM sampling timelines (reveals tail effect, pipeline bubbles); `--list` shows a report's series |
| `list_flashinfer_workloads.py` | Browse a flashinfer-trace dataset: show axes, histogram workload shapes, print safetensors paths for specific UUIDs |

### Typical Python workflow

```bash
export FIB_DATASET_PATH=<flashinfer-trace dataset root>  # if using FIB workloads

# (Optional) Browse workload shapes for a flashinfer-trace dataset
python3 "$HELPERS/list_flashinfer_workloads.py" --definition <def_name>
python3 "$HELPERS/list_flashinfer_workloads.py" --definition <def_name> --unique-axes <axis1>,<axis2> --no-paths

# Key metrics and NCU rule results for each report (+ a side-by-side comparison)
python3 "$HELPERS/analyze_reports.py" --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep" --tag <tag1> \
    --report "$PROFILE_RUN_DIR/reports/full_<tag2>.ncu-rep" --tag <tag2>

# Per-line stall hotspots (--set full has the source counters)
python3 "$HELPERS/extract_stall_hotspots.py" --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep" --tag <tag1> \
    [--binary path/to/the/profiled/binary_or_lib.so]

# ASCII PM timeline plots
python3 "$HELPERS/plot_timeline.py" --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep" --tag <tag1>
```

The three report scripts take `--run-dir` and write under `<run-dir>/analysis/`. A report holding several kernels gives one output per kernel (`<tag>.<i>_<function>`); `--kernel REGEX` keeps the matching ones. None of them needs a GPU.

`ncu_utils.py` finds `ncu_report` from the `ncu` on PATH, the way the CUDA toolkit's `ncu` launcher finds Nsight Compute. If `ncu` is not on PATH, put `<Nsight Compute install>/extras/python` on `PYTHONPATH`.
