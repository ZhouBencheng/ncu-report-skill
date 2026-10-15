# Tool usage

The `scripts/` directory contains executable Python tools and their shared module. The `assets/` directory contains CUDA / C++ resources to copy into profiling harnesses. See [`../SKILL.md`](../SKILL.md) for context.

## Assets: C++ / CUDA

| File | Purpose |
|---|---|
| [`harness_template.cu`](../assets/harness_template.cu) | Starting point for a profiling harness. Copy into your run dir, fill in the `TODO(you)` sections. |
| [`safetensors_loader.h`](../assets/safetensors_loader.h) | Header-only safetensors reader (no external deps). Use from your harness to load real workload tensors. |

### Typical harness setup

```bash
cd profile/<run_name>/harness/
cp /path/to/skills/ncu-report-skill/assets/harness_template.cu my_kernel_harness.cu
cp /path/to/skills/ncu-report-skill/assets/safetensors_loader.h .
# edit my_kernel_harness.cu to include your kernel + fill in main()
# Thor (SM110): -gencode=arch=compute_110,code=sm_110
# B200  (unchanged): -gencode=arch=compute_100,code=sm_100
nvcc -O2 -std=c++17 -lineinfo -gencode=arch=compute_110,code=sm_110 \
     my_kernel_harness.cu -o my_kernel_harness
```

## Scripts: Python

| File | Purpose |
|---|---|
| [`ncu_utils.py`](../scripts/ncu_utils.py) | Shared helpers: `load_report`, `safe`, `per_pc_values`, `B200_KEY_METRICS`, `rule_speedups`, ... |
| [`analyze_reports.py`](../scripts/analyze_reports.py) | Extract key metrics + side-by-side comparison from one or more `.ncu-rep`s |
| [`extract_stall_hotspots.py`](../scripts/extract_stall_hotspots.py) | Aggregate per-PC stall samples → per-source-line rankings (requires source-level report) |
| [`plot_timeline.py`](../scripts/plot_timeline.py) | ASCII plot PM sampling timelines (reveals tail effect, pipeline bubbles) |
| [`list_flashinfer_workloads.py`](../scripts/list_flashinfer_workloads.py) | Browse a flashinfer-trace dataset: show axes, histogram workload shapes, print safetensors paths for specific UUIDs |

### Typical Python workflow

```bash
export PROFILE_RUN_DIR=profile/<run_name>
SCRIPTS=/path/to/skills/ncu-report-skill/scripts
export FIB_DATASET_PATH=/path/to/flashinfer-trace  # if using FIB workloads

# (Optional) Browse workload shapes for a flashinfer-trace dataset
python3 $SCRIPTS/list_flashinfer_workloads.py --definition <def_name>
python3 $SCRIPTS/list_flashinfer_workloads.py --definition <def_name> --unique-axes <axis1>,<axis2> --no-paths

# Extract key metrics for each report
python3 $SCRIPTS/analyze_reports.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep --tag <tag1> \
    --report $PROFILE_RUN_DIR/reports/full_<tag2>.ncu-rep --tag <tag2>

# Per-line stall hotspots (requires source-level reports, collected with --set source)
python3 $SCRIPTS/extract_stall_hotspots.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/source_<tag1>.ncu-rep --tag <tag1> \
    --report $PROFILE_RUN_DIR/reports/source_<tag2>.ncu-rep --tag <tag2>

# ASCII PM timeline plots
python3 $SCRIPTS/plot_timeline.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/full_<tag1>.ncu-rep --tag <tag1> \
    --report $PROFILE_RUN_DIR/reports/full_<tag2>.ncu-rep --tag <tag2>
```

All three scripts take `--run-dir` and write under `<run-dir>/analysis/`.

`ncu_utils.py` tries to auto-locate `ncu_report` from common CUDA install paths. If that fails, set `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:/opt/nvidia/nsight-compute/2026.1.1/extras/python
```
