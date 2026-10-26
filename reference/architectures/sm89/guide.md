# Ada / RTX 4090 / SM89

RTX 4090 runtime CC is **8.9**. Use native `sm_89` for ordinary code when supported by the selected compiler. Record actual SM count, clocks, memory bandwidth assumptions and driver/toolkit versions; these are device facts, not constants shared by every Ada GPU.

## Collection and interpretation

Query the installed NCU before selecting sections or clock options. A CUDA 12.x-era collector does not necessarily have the PM layout, timed warp API, or `boost` option used by NCU 2026.3. The common metric list is a candidate list; preserve report absence and raw units. Inspect the actual `--clock-control` default and observed frequency before an A/B comparison.

Ada supports Tensor Core paths, but SM100 UTC/TMEM counters are not its Tensor Core utilization measure. Absence of `utcmma`, `utchmma` or TMEM metrics is expected and must not produce a "Tensor Cores idle" diagnosis. Use available Tensor/HMMA counters together with emitted instructions and workload dtype.

## Applicable optimization paths

- Warp-level MMA and `cp.async` pipelines are available subject to exact instruction/dtype constraints.
- TMA, WGMMA, tcgen05, TMEM and CLC are not the SM89 execution path. Use shared memory/registers and software scheduling where the common diagnosis calls for buffering or persistent workers.
- Re-measure vector widths, tile shapes, register budgets and pipeline stages. Their legality and profitability are separate questions.
- Compare memory access efficiency to the instruction width and active lanes; a B200 utilization threshold or speedup is not Ada evidence.

Sources: [GPU CC](https://developer.nvidia.com/cuda/gpus), [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [NCU CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html). Actual validation scope is in [validation](../../validation.md).
