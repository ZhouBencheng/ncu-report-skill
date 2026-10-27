# B200 / SM100

B200 runtime CC is **10.0**. Keep the compile target separate: code requiring architecture-specific instructions may need `sm_100a`; ordinary code and family-specific features have different targets. Confirm against the selected compiler and the exact instruction rather than adding a suffix to every CC.

The existing B200/B300 metric roster remains available through `KEY_METRICS` and `B200_KEY_METRICS`. Automatic report routing selects SM100 from device CC/name and preserves the original raw metric names. The extra UTC/TMEM candidates live in `profile.json`; shared metric names live once in `reference/common-metrics.json`.

Read only the relevant detail:

- [Metric reference](metric-reference.md): retained NCU 2026.1/2026.3 names and semantics.
- [Collection recipes](collection.md): retained B200/B300 recipes; query the installed collector before selecting flags.
- [Programming reference](programming.md): B200-specific configurations and examples. Reported throughput and hardware values remain scoped to their stated setup.

Generic diagnosis is in [the common playbook](../../06-diagnosis-playbook.md). For low Tensor utilization, distinguish MMA execution, TMEM traffic and pipeline waits; neither the TMEM counter nor an instruction count alone measures useful matrix work. `tcgen05.cp` copies shared memory to TMEM and is not a substitute for global-to-shared TMA or `cp.async`.

No B200 device was supplied for the new adaptation campaign. Historical offline regression checked original metric values/order and parsing behavior; its sources remain in Git history. That does not claim a new B200 hardware run. See [validation](../../validation.md).

Sources: [GPU CC](https://developer.nvidia.com/cuda/gpus), [Blackwell tuning](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html), [PTX target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
