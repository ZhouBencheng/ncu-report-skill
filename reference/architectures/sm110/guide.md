# Thor / SM110

Jetson AGX Thor (T5000/T4000) is **CC 11.0 / SM110**, not SM120. Record actual runtime resources and the selected compiler; do not copy B200's SM count, HBM system or tuning parameters.

## Collection and reading

Use the collector supported by the installed JetPack/L4T stack and record collector/reader versions separately. Query sections, sets and metrics. A `full` report can have different PM series from NCU 2026.3; missing timed-warp APIs or PM data must leave a bounded static analysis usable.

Use platform-aware clocks: preflight proposes `--clock-control none` and never changes power/clock settings. Record L4T, power mode, frequency and thermal conditions. Counter access may require an approved privileged invocation. Do not use a desktop `nvidia-smi` lock recipe as a universal Jetson instruction.

## Applicable optimization paths

Thor has tcgen05/TMEM instruction support for documented target variants, so its profile shares the **candidate names** used for those observations with SM100. The actual collector inventory/report decides which candidates are readable. Orin and Thor share a small MCC candidate list; read/write throughput percentages keep their original names and are not substituted for DRAM bytes. This does not assert that a B200 kernel binary, 2-CTA layout, tile, dtype form or resource budget is valid on Thor.

Distinguish `sm_110` runtime family selection from the compiler's exact `sm_110a`/`sm_110f` requirements. Check each instruction's target notes and verify compilation, numerical correctness and performance independently. Reuse general buffering, ownership, fusion and scheduling principles; keep concrete kernel implementations and measured benefits target-specific.

`tcgen05.cp` is a shared-memory-to-TMEM operation, not a global-memory load. A latency diagnosis still needs the actual memory path and synchronization evidence.

Sources: [GPU CC](https://developer.nvidia.com/cuda/gpus), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [NCU clock control](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control). Actual execution/collection evidence is in [validation](../../validation.md).
