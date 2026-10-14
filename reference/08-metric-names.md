# Metric Name Reference (Blackwell, ncu 2026.1)

Metric names change between GPU generations and Nsight Compute releases, and many older docs use names that `--set full` does not collect on Blackwell. This doc lists the names a `--set full` report holds with Nsight Compute 2026.1 on B200 / B300 (sm_100 / sm_103), and what to use instead of the older names.

If a metric returns `None` on your kernel, first check this doc, then enumerate the names the report has:

```python
action.metric_names()
```

A name missing from a report may still exist: no section collects it. Ask for it with `--metrics name1,name2` (`03-collection.md`, Recipe 5).

---

## Names older docs use

| Older name | In a `--set full` report (ncu 2026.1) |
|---|---|
| `smsp__inst_executed_op_global_ld.sum` | use **`smsp__sass_inst_executed_op_global_ld.sum`**; the old name exists, collect it with `--metrics` |
| `smsp__inst_executed_op_global_st.sum` | **`smsp__sass_inst_executed_op_global_st.sum`** |
| `smsp__inst_executed_op_local_ld.sum` | **`smsp__sass_inst_executed_op_local_ld.sum`** |
| `smsp__inst_executed_op_local_st.sum` | **`smsp__sass_inst_executed_op_local_st.sum`** |
| `smsp__inst_executed_op_shared_ld.sum` | **`smsp__sass_inst_executed_op_shared_ld.sum`** |
| `smsp__inst_executed_op_shared_st.sum` | **`smsp__sass_inst_executed_op_shared_st.sum`** |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | not collected: compute `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`, or request it with `--metrics` |
| `dram__bytes.sum` | not collected: use `dram__bytes_read.sum + dram__bytes_write.sum`, or request it with `--metrics` |
| `sm__inst_executed_pipe_fmaheavy.*` | not collected: `sm__inst_executed_pipe_fma.*` is; request it with `--metrics` |
| `smsp__warps_issue_stalled_<reason>_per_issue_active.pct` | **`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`** (note `average_` prefix and `ratio` suffix) |
| `pmsampling:sm__throughput.*`, `pmsampling:dram__throughput.*`, `pmsampling:sm__warps_active.*`, `pmsampling:l1tex__throughput.*` | the PmSampling section's series are `<UNIT>.<group>.<metric>`: see "PM sampling" below |

---

## Curated metric set

A `--set full` report holds these on B200 / B300 with Nsight Compute 2026.1. Always verify for your specific ncu version by enumerating with `action.metric_names()` — NVIDIA occasionally renames metrics between releases.

### Launch geometry / occupancy
```
launch__grid_size
launch__block_size
launch__grid_dim_x, launch__grid_dim_y, launch__grid_dim_z
launch__block_dim_x, launch__block_dim_y, launch__block_dim_z
launch__thread_count
launch__waves_per_multiprocessor
launch__registers_per_thread
launch__shared_mem_per_block
launch__shared_mem_per_block_static
launch__shared_mem_per_block_dynamic
launch__occupancy_limit_blocks
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__occupancy_limit_warps
device__attribute_multiprocessor_count
device__attribute_max_warps_per_multiprocessor
sm__maximum_warps_per_active_cycle_pct              # theoretical occupancy %
```

### SOL (Speed-of-Light) / throughput
```
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed
l1tex__throughput.avg.pct_of_peak_sustained_active
lts__throughput.avg.pct_of_peak_sustained_elapsed
dram__bytes_read.sum
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_read.sum.per_second                       # achieved BW
dram__bytes_write.sum
dram__bytes_write.sum.pct_of_peak_sustained_elapsed
dram__sectors_read.sum
dram__sectors_write.sum
```

### Timing and clock
```
gpu__time_duration.sum                                # units: ns (check .unit())
sm__cycles_elapsed.avg.per_second                     # the SM clock the kernel ran at (see --clock-control)
smsp__cycles_active.avg
smsp__issue_active.avg.per_cycle_active               # issue rate per scheduler
smsp__issue_active.avg.pct_of_peak_sustained_active
```

### Warp activity
```
sm__warps_active.avg.pct_of_peak_sustained_active     # achieved occupancy %
sm__warps_active.avg.per_cycle_active
sm__warps_active.max.per_cycle_active
sm__warps_active.min.per_cycle_active
smsp__warps_active.avg.per_cycle_active               # per sub-partition
smsp__warps_eligible.avg.per_cycle_active
smsp__warps_eligible.max.per_cycle_active
```

### Compute pipelines
```
sm__inst_executed.avg.per_cycle_active                # IPC
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active
sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active

sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_subpipe_imma_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_subpipe_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed    # absent on B300 (no FP64 MMA)
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.avg   # raw BF16→FP32 tensor ops
```

### Cache hit rates
```
l1tex__t_sector_hit_rate.pct                                       # overall L1
lts__t_sector_hit_rate.pct                                         # overall L2
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct             # L1 hit on global loads
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct             # L1 hit on global stores
```

### Memory access counts & sectors
```
smsp__sass_inst_executed_op_global_ld.sum                          # global LD instruction count
smsp__sass_inst_executed_op_global_st.sum                          # global ST count
smsp__sass_inst_executed_op_local_ld.sum                           # local LD (register spill)
smsp__sass_inst_executed_op_local_st.sum                           # local ST (register spill)
smsp__sass_inst_executed_op_shared.sum                             # total shared mem ops
smsp__sass_inst_executed_op_shared_ld.sum
smsp__sass_inst_executed_op_shared_st.sum

l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                     # L1 sectors for global LD
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                     # L1 sectors for global ST
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                    # LD request count
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                    # ST request count
# sectors/request = sectors.sum / requests.sum (ideal = 4 for 128B coalesced)
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio    # store efficiency (max 32)
```

### Stall reasons — aggregate ratios
```
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio
smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio
smsp__average_warps_issue_stalled_misc_per_issue_active.ratio
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio       # productive (= 1.0)
```

### Stall reasons — per-PC (SourceCounters section; in `--set full`)
```
smsp__pcsamp_sample_count                                           # total sample count
smsp__pcsamp_warps_issue_stalled_long_scoreboard                    # per-PC counts
smsp__pcsamp_warps_issue_stalled_short_scoreboard
smsp__pcsamp_warps_issue_stalled_wait
smsp__pcsamp_warps_issue_stalled_barrier
smsp__pcsamp_warps_issue_stalled_math_pipe_throttle
smsp__pcsamp_warps_issue_stalled_mio_throttle
smsp__pcsamp_warps_issue_stalled_lg_throttle
smsp__pcsamp_warps_issue_stalled_tex_throttle
smsp__pcsamp_warps_issue_stalled_not_selected
smsp__pcsamp_warps_issue_stalled_dispatch_stall
smsp__pcsamp_warps_issue_stalled_drain
smsp__pcsamp_warps_issue_stalled_no_instructions
smsp__pcsamp_warps_issue_stalled_selected
smsp__pcsamp_warps_issue_stalled_branch_resolving
smsp__pcsamp_warps_issue_stalled_membar
smsp__pcsamp_warps_issue_stalled_sleeping
smsp__pcsamp_warps_issue_stalled_misc
smsp__pcsamp_warps_issue_stalled_tex_throttle
smsp__pcsamp_warps_issue_stalled_<reason>_not_issued                # samples where the warp's scheduler issued nothing
```

Each of these has `num_instances() > 0` with `correlation_ids()` that map to PCs. Use `action.source_info(pc)` to map PCs to `(file, line)` (innermost inlined frame; see `04-python-api.md`).

### PM sampling (time series)

The PmSampling section (in `--set full`) gives `<UNIT>.<group>.<metric>` series, e.g.:
```
TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed
TPC.TriageCompute.sm__inst_executed_realtime.avg.per_cycle_active
TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
TPC.TriageCompute.sm__pipe_fma_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed
FBSP.TriageCompute.dram__throughput.avg.pct_of_peak_sustained_elapsed
LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed
SM_A.TriageCompute.l1tex__throughput.avg.pct_of_peak_sustained_elapsed
```

The PmSampling_WarpStates section (`--section PmSampling_WarpStates`) gives the stall-reason series:
```
pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_short_scoreboard.avg
pmsampling:smsp__warps_issue_stalled_wait.avg
pmsampling:smsp__warps_issue_stalled_barrier.avg
pmsampling:smsp__warps_issue_stalled_sleeping.avg
pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg
pmsampling:smsp__warps_issue_stalled_mio_throttle.avg
pmsampling:smsp__warps_issue_stalled_lg_throttle.avg
...                                                  # one per stall reason
```

`plot_timeline.py --list` prints every series a report has. The sample interval was ~1 µs on a B300: a kernel shorter than ~20 µs gives few samples.

---

## Discovering metrics for new GPUs

```bash
# All metrics of the GPU in this machine (needs the GPU)
ncu --query-metrics
ncu --query-metrics | grep -i issue_stalled

# Without the GPU: name the chip; ncu --list-chips lists the names
ncu --list-chips
ncu --query-metrics --chip <chip>
```

From Python, you can also enumerate per-report:
```python
all_names = action.metric_names()      # all metrics collected in this report
```

---

## Gotchas

1. **Metric exists in ncu's list but returns `None` from Python**: the metric wasn't *collected* in this report. Rerun ncu with the right `--section` or `--set`.
2. **Metric value is `0.0`**: either the hardware counter reports zero (e.g., no tensor core activity), or the metric is synthetic and depends on other metrics that weren't collected.
3. **`.avg` vs `.sum` vs `.max`**: each aggregate is a separate metric name. `.avg` is most commonly useful for rates / percentages; `.sum` for counts; `.max` for worst-case analysis.
4. **`pct_of_peak_sustained_elapsed` vs `pct_of_peak_sustained_active`**: `_elapsed` normalizes against total kernel time (including idle SMs); `_active` normalizes against cycles where the SM was actually running. `_elapsed` is more honest for under-utilized kernels.
