# ncu-report-skill
> [!IMPORTANT]
> This skill is maintained as a standalone submodule of
> [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
> for easy installation.
>
> For bug reports, feature requests, and discussions, please use the main KDA repository:
> https://github.com/mit-han-lab/kernel-design-agents

A Claude Code skill for profiling CUDA kernels with Nsight Compute with target-aware guidance for B200/B300, RTX 4090, Jetson Orin and Thor. Covers the full workflow: get a profile target (the real code, or a standalone harness), run `ncu`, parse reports with the Python API, walk through six analysis dimensions, match patterns to a diagnosis playbook, and write an evidence-backed optimization report.

Written against Nsight Compute 2026.3.1, the newest release (CUDA 13.2 bundles 2026.1.1; the skill's commands and helpers handle both): its sets and sections, report format, rule-result API, PM-sampling series and timelines, and default clock control.

The skill is self-contained: reference docs, reusable helper scripts (harness template, safetensors loader, report-analysis Python), and a companion Blackwell programming reference all ship in this repo.

---

## What's in this repo

```
.
├── SKILL.md                          ← skill entry point (with YAML frontmatter)
├── helpers/                          ← reusable code
│   ├── harness_template.cu           ← standalone profiling harness template
│   ├── safetensors_loader.h          ← header-only safetensors reader (no deps)
│   ├── list_flashinfer_workloads.py  ← browse flashinfer-trace datasets
│   ├── analyze_reports.py            ← key metrics, NCU rule results, comparisons from .ncu-rep files
│   ├── extract_stall_hotspots.py     ← per-line stall aggregation (inline-aware with --binary)
│   ├── plot_timeline.py              ← ASCII PM-sampling timeline plots (reveals tail effects)
│   ├── ncu_utils.py                  ← shared Python helpers: finds ncu_report, key metric list, rules, PM series
│   └── README.md
├── reference/                        ← detailed reference docs
│   ├── 00-directory-layout.md        ← profile directory conventions (read first)
│   ├── 01-workflow.md                ← end-to-end profiling checklist
│   ├── 02-harness-guide.md           ← profiling the real code or a standalone harness
│   ├── 03-collection.md              ← ncu command recipes
│   ├── 04-python-api.md              ← ncu_report Python API patterns
│   ├── 05-analysis-dimensions.md     ← six analysis dimensions
│   ├── 06-diagnosis-playbook.md      ← pattern → cause → fix
│   ├── 07-report-template.md         ← final report structure
│   ├── 08-metric-names.md            ← shared metric-state and normalization contract
│   └── 09-common-issues.md           ← permissions, PM sampling, JIT toolchains, etc.
└── blackwell-cuda-programming.md     ← compatibility link to reference/architectures/sm100/programming.md
```

---

## Installation

Claude Code discovers skills in two locations:

- `~/.claude/skills/<skill_name>/SKILL.md` — **user-level**, available in every project
- `<repo>/.claude/skills/<skill_name>/SKILL.md` — **project-level**, scoped to one repo

Three ways to install this skill:

### Option 1 — Symlink from a clone (recommended)

Keeps the skill version-controlled and easy to update; edits in the clone are picked up instantly.

```bash
# Clone somewhere stable
git clone https://github.com/mit-han-lab/ncu-report-skill.git ~/workspace/ncu-report-skill

# User-level install: make the skill available in every project
mkdir -p ~/.claude/skills
ln -s ~/workspace/ncu-report-skill ~/.claude/skills/ncu-report-skill

# Or project-level install: scope to one repo
cd /path/to/other-repo
mkdir -p .claude/skills
ln -s ~/workspace/ncu-report-skill .claude/skills/ncu-report-skill
```

Pull updates with `cd ~/workspace/ncu-report-skill && git pull`. The symlinks pick up the new content automatically.

### Option 2 — Copy into place

If you prefer a static copy over a symlink:

```bash
git clone https://github.com/mit-han-lab/ncu-report-skill.git /tmp/ncu
mkdir -p ~/.claude/skills
cp -r /tmp/ncu ~/.claude/skills/ncu-report-skill
```

### Option 3 — Git submodule (for a project-level install committed alongside the repo)

```bash
cd /path/to/other-repo
git submodule add https://github.com/mit-han-lab/ncu-report-skill.git .claude/skills/ncu-report-skill
git commit -m "Add ncu-report-skill as a submodule"
```

---

## How Claude uses this skill

Once installed at `~/.claude/skills/ncu-report-skill/` (or project-level), Claude Code will:

1. Advertise the skill's name + description in the system reminder of new conversations.
2. Let the user invoke it manually via `/ncu-report-skill` or let the model invoke it with the Skill tool when the conversation matches the `description` triggers.

When invoked, Claude reads `SKILL.md`, follows its workflow (phases 0 → 6 in `reference/01-workflow.md`), and uses the helper scripts in `helpers/` as needed.

---

## Running the helpers directly (no Claude needed)

The Python helpers work standalone for any `.ncu-rep` / `.ncu-repz` you have:

```bash
# The helpers find ncu_report from the ncu on PATH; this prints where
python3 ~/.claude/skills/ncu-report-skill/helpers/ncu_utils.py

# The run directory holding reports/
export RUN=<your profile root>/myrun

# Extract key metrics from one or more reports
python3 ~/.claude/skills/ncu-report-skill/helpers/analyze_reports.py \
    --run-dir "$RUN" \
    --report "$RUN/reports/full_<tag>.ncu-rep" --tag <tag>

# Per-line stall hotspots (when SourceCounters are present; --binary for inlined code)
python3 ~/.claude/skills/ncu-report-skill/helpers/extract_stall_hotspots.py \
    --run-dir "$RUN" \
    --report "$RUN/reports/full_<tag>.ncu-rep" --tag <tag>

# ASCII PM-sampling timelines
python3 ~/.claude/skills/ncu-report-skill/helpers/plot_timeline.py \
    --run-dir "$RUN" \
    --report "$RUN/reports/full_<tag>.ncu-rep" --tag <tag>

# Browse a flashinfer-trace dataset to pick workload shapes
export FIB_DATASET_PATH=/path/to/flashinfer-trace
python3 ~/.claude/skills/ncu-report-skill/helpers/list_flashinfer_workloads.py \
    --definition <your_definition_name>
```

The C++ harness template + safetensors loader live under `helpers/`; copy them into your profile run's `harness/` directory and fill in the kernel body. See `reference/02-harness-guide.md` for details.

---

## Requirements

- CUDA Toolkit with `nvcc`, `cuobjdump`, `nvdisasm` (tested with 13.2)
- Nsight Compute CLI `ncu` (tested with 2026.3.1 and 2026.1.1)
- The `ncu_report` Python module (ships with Nsight Compute under `extras/python/`)
- An NVIDIA GPU with permission to access performance counters (see `reference/09-common-issues.md` if `ncu` reports `ERR_NVGPUCTRPERM`)

The analyzer routes by each report's observed GPU identity and uses common plus selected-architecture candidate metrics. Unknown identity uses generic analysis, never B200 defaults. Collector availability and report coverage remain separate from architecture routing; see [metric semantics](reference/08-metric-names.md) and [validated environments](reference/validation.md).

---

## License

MIT

## Target-specific loading

The root skill loads a shared workflow and one relevant guide under `reference/architectures/`. SM100 programming/collection references remain there as a single source; SM103/SM110 reuse metric candidates where applicable without inheriting device limits or binary compatibility. `helpers/preflight.py` discovers the actual collector, metric inventory and sections before collection.

```bash
python3 helpers/preflight.py --ncu /actual/path/to/ncu --output "$PROFILE_RUN_DIR/preflight.json"
python3 helpers/analyze_reports.py --run-dir "$PROFILE_RUN_DIR" --report report.ncu-rep --tag case
python3 -m unittest discover -s tests -v
```

No GPU is needed for the unit tests. Real-device evidence and its limits are recorded in [validation](reference/validation.md).
