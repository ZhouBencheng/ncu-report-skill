# B300 / SM103

B300 runtime CC is **10.3**. Keep its compiled target (`sm_103`, `sm_103a`, or an applicable family target) separate from runtime identity and validate required instructions with the installed compiler.

The existing B300 workflow uses the [shared B200/B300 metric reference](../sm100/metric-reference.md) and [version-scoped collection recipes](../sm100/collection.md). Its profile deliberately shares the metric candidate data with SM100; this is not inheritance of all hardware limits, tuning parameters or executable compatibility.

Use actual B300 resources, clocks and workload measurements. An SM100-only cubin or a B200 benchmark is not B300 execution evidence. This change preserves existing B300 routing/metric coverage; no new B300 hardware run is claimed. See [validation](../../validation.md).

Sources: [GPU CC](https://developer.nvidia.com/cuda/gpus), [feature-set targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#feature-set-compiler-targets).
