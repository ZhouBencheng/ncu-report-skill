# Orin / SM87

Runtime CC is **8.7**; compile ordinary native code for `sm_87`, using a compiler that lists this target. Do not substitute `sm_80` or `sm_86` for a native Orin build. Instructions and shapes still need their own compiler/PTX checks.

## Collection and reading

Use the JetPack/L4T-compatible collector installed on the device. Record its version separately from the reader; installing a new Python reader does not upgrade the collector. Query sections and metrics before copying recipes from desktop NCU. If the report reader shipped with a collector cannot read a report in Python, use a compatible reader in a separate environment and record that pairing.

Jetson is an integrated-memory platform. Read actual device resources and available memory-controller metrics; do not substitute B200 HBM bandwidth or assume every `dram__*` metric exists. The common candidate list is a starting point, not proof of availability. The verified collector inventory includes `mcc__dram_throughput_op_read/write.avg.pct_of_peak_sustained_elapsed`; these candidate readings remain under their original names and units, and are not aliases for DRAM bytes or a bandwidth measurement. They may need explicit collection.

Use `--clock-control none` when NCU cannot control platform clocks. Record L4T, power mode and observed clocks; platform tools such as `jetson_clocks` are a separate experiment setting, not an automatic preflight action. Counter permission can require an approved privileged collector invocation. Missing PM sampling is an evidence gap, not zero activity.

## Applicable optimization paths

- Ordinary `cp.async` global-to-shared pipelines and warp-level MMA are candidates, subject to the instruction's dtype/shape requirements.
- TMA, WGMMA, tcgen05, TMEM and CLC are not Orin execution paths. Do not recommend them to fix a stall just because a generic Blackwell example does.
- For small GEMV/decode or reductions, compare launch overhead, memory traffic, available parallel work and register pressure. Tensor Core peak numbers alone do not select the best path.
- Shared-memory buffering, coalescing, fusion and software scheduling use the common diagnosis principles; tune their implementation and stages on this device.

Sources: [GPU CC](https://developer.nvidia.com/cuda/gpus), [PTX target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [NCU support](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#support), [clock control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control). Execution evidence is recorded separately in [validation](../../validation.md).
