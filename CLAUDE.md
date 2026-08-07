# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EasyBFE is a Python package that prepares, runs, and analyzes protein-ligand binding free energy
calculations (ABFE and RBFE) using AMBER/pmemd with Hamiltonian replica-exchange (H-REMD) alchemical
MD and MBAR free energy estimation. It ships as a CLI (`easybfe`) built with `rich_click`.

- **ABFE** (absolute): three legs — `solvent`, `complex`, `restraint` (Boresch orientational
  restraints). ΔG_bind = ΔG_complex − ΔG_solvent + ΔG_restraint + ΔG_boresch.
- **RBFE** (relative): two legs — `solvent`, `complex` (alchemical A→B mutation). ΔΔG = ΔG_complex − ΔG_solvent.

For end-user CLI usage (config schemas, full option tables, network algorithms, constrained docking),
see [skills/easybfe/SKILL.md](skills/easybfe/SKILL.md) and its `reference/` docs — that skill is the
canonical CLI usage reference and should be consulted before explaining CLI behavior to a user.

## Environment

This project runs on NERSC Perlmutter. Before running Python/pytest in a fresh shell:

```bash
source ~/env/easybfe.env
```

This activates the `easybfe` conda environment (see `environment.yml`) and loads GPU toolchain modules
(gcc, cudatoolkit, openmpi, pmemd) needed for MD-dependent tests. If a command fails with missing
packages or the wrong Python, confirm this was sourced in the current shell session.

### Getting a compute node

Compute-heavy work (compilation, MD simulation, large test runs) must not run on the login node.
Allocate a node first; account is `m2834`.

```bash
# Interactive session (qos interactive, up to 4 h) — the usual choice.
# The user has these as shell aliases `gpu` and `cpu`:
salloc --nodes 1 --qos interactive --time 4:00:00 --constraint gpu --account=m2834   # GPU node
salloc --nodes 1 --qos interactive --time 4:00:00 --constraint cpu --account=m2834   # CPU node

# One-off command without an interactive shell:
srun --nodes 1 --qos interactive --time 4:00:00 --constraint gpu --account=m2834 <command>

# Short validation runs (qos debug, max 30 min) — faster to schedule:
srun --nodes 1 --qos debug --time 0:30:00 --constraint gpu --account=m2834 <command>

# Longer work (qos premium, up to 2 days) — request only the time actually needed:
srun --nodes 1 --qos premium --time 4:00:00 --constraint gpu --account=m2834 <command>
```

Use `--constraint gpu` for anything touching `pmemd.cuda` / OpenMM GPU paths, `--constraint cpu` for
parameterization, analysis, and CPU-only MD (e.g. `pmemd.MPI`). Re-`source ~/env/easybfe.env` inside
the allocation — the module/conda state does not carry over from the login shell.

Plain `pytest tests/` (unit tests, no MD) is fine on the login node.

## Common commands

```bash
# Install in editable mode (after sourcing the env)
pip install -e .

# Run the full test suite
pytest tests/

# Run a single test file / test
pytest tests/test_boresch.py
pytest tests/test_boresch.py::test_some_function -v

# CLI help
easybfe --help
easybfe COMMAND --help
easybfe COMMAND SUBCOMMAND --help
```

Test output directories under `tests/` prefixed with `_test_*` or `_e2e_*` are artifacts from previous
test runs (e.g. `_test_ligand_abfe`, `_e2e_abfe_pipeline`), not fixtures — real fixture data lives in
`tests/data/`.

## Architecture

### CLI → workflow → registry layering

`easybfe/cli/main.py` wires up top-level `click` groups (`abfe`, `rbfe`, `ligand`, `protein`, `md`),
each defined in its own `easybfe/cli/*.py`. CLI commands parse args/config and delegate to workflow
classes; they contain no simulation logic themselves.

Several subsystems are pluggable via a shared generic `Registry` (`easybfe/core/registry.py`):
implementations self-register with a decorator when their module is imported, and are looked up by
string name from config.

- `easybfe/smff/registry.py` → `PARAMETRIZER_REGISTRY` — small-molecule force field parameterizers
  (`gaff.py` = GAFF/acpype, `openff.py`, `custom.py`), selected via the `-f/--forcefield` /
  `--engine` CLI flags or config.
- `easybfe/mapping/registry.py` → `MAPPER_REGISTRY` — RBFE atom mappers (`lazymcs.py`, `kartograf.py`,
  `lomap.py`).
- `easybfe/network/registry.py` — RBFE perturbation network algorithms (wraps `openfe`).
- `easybfe/boresch/finders.py` / `md_finder.py` → `BORESCH_FINDER_REGISTRY` — Boresch restraint
  selection algorithms (static-pose `RxRx...Finder` vs. MD-trajectory-based
  `RxRxMDBoreschRestraintsFinder`, the latter driven by `boresch_md:` config and used when
  `boresch.algorithm: rxrx-md`).

When adding a new implementation of one of these, register it in its module (decorator pattern) and
ensure that module is imported somewhere reachable from package init so registration actually runs.

### Core data models (`easybfe/core/`)

Pydantic models used across the codebase:
- `Ligand` (`ligand.py`) — name, SMILES, 3D mol block, and `auxiliary_files` (a dict of generated file
  contents such as prmtop/inpcrd/itp, keyed by filename) — this is how parameterized ligand output
  directories are represented/serialized.
- `Protein` (`protein.py`).
- `LigandPerturbation` (`perturbation.py`) — an A→B RBFE edge: atom mapping plus per-leg ΔΔG results
  (solvent/complex/gas) and their standard deviations.
- `sql_models.py` — SQL persistence layer for the above (used for network/campaign bookkeeping).

### Config (`easybfe/config/`)

Pydantic config schemas validated from YAML/JSON (`read_file`). Key schemas: `AmberAbfeConfig`
(`easybfe/abfe/config.py`), `AmberLigandRbfeConfig` (RBFE setup), plus shared `setup.py`,
`analysis.py`, `protein_prep.py`. `easybfe/config/amber/`
holds AMBER-specific mdin/step templates (`AmberMdin`, `AmberStepConfig`, consumed by
`easybfe/amber/workflow.py`'s `Step` class, which renders `.in` files and the `pmemd` command line for
each MD stage). Generating `#SBATCH` submission scripts is not implemented — a leg is submitted by
pointing `sbatch` at a hand-written wrapper around its `run.sh`.

### Leg setup and execution (`easybfe/amber/`)

- `prep_ligand_abfe.py` / `prep_ligand_rbfe.py` / `prep_plain_md.py` — build AMBER topology/coordinates
  for each leg type from ligand + protein inputs.
- `workflow.py` — `Step` wraps one `pmemd` invocation (writes `<name>.in`, renders the command line
  two ways: a readable multi-line `<stage>.sh` for debugging one lambda window by hand, and a
  single-line groupfile entry). `create_script_for_workflows` writes one `run.sh` per leg covering
  every stage; select stages with `run.sh [--from STAGE] [--until STAGE] [--force] [--list]`.

  **Leg-script contract**: the generated scripts must depend on bash + AMBER only — never on easybfe
  or python. easybfe writes them (setup) and reads their status files (analyze); it is not needed to
  execute them, so a leg can run in a plain AMBER image. `tests/test_leg_script.py` enforces this.

  The script owns the tag state machine and resumes at the first stage lacking a `<stage>.done.tag`:
  `<stage>.done.tag` per stage, `preprod.done.tag` after the second-to-last stage (early-stop phase),
  `done.tag` after the last, plus leg-level `running`/`error`/`killed.tag`. It also writes
  `status.json` (state, failing stage, exit code, error excerpt) — the only writer; python reads it.

### Analysis (`easybfe/analysis/`)

MBAR-based free energy estimation (`mbar.py`, `mle.py`) consumes per-window energies from completed
legs; `abfe.py`/`rbfe.py` orchestrate per-workflow-type analysis and write `result.json`.
`boresch.py`/`trajectory.py`/`plain_md.py` support representative-frame and restraint-geometry analysis
for MD-based Boresch restraint selection. `plot.py` produces convergence/energy plots.

### One-line ABFE pipeline (`easybfe/abfe/piepline.py`, note the filename typo)

`ABFE` (in `piepline.py`) is a single entry point chaining: ligand parameterization → (if
`boresch.algorithm: rxrx-md`) plain protein-ligand MD + representative-frame/anchor selection → leg
setup → local blocking execution of each leg's `run.sh` → analysis. Writes everything under one output
directory (`<dir>/ligand/`, `<dir>/boresch-md/`, `<dir>/abfe/{solvent,complex,restraint}` +
`boresch.dat` + `result.json`, `<dir>/abfe.log`). Supports early-stopping: after pre-production stages
(`04.pre_prod`) it estimates ΔG and skips the expensive `05.prod` stage if the estimate is weaker than
`early_stop_threshold`. This is distinct from the multi-node `setup`/submit/`analyze` workflow, which is
preferred when legs need to be distributed across separate Slurm jobs/nodes.

## Notes

- `easybfe/abfe/piepline.py` is misspelled (missing the second `e` in "pipeline") — this is the actual
  module name on disk, not a typo to silently "fix" in unrelated edits (it's part of the public import
  path).
- Docking backends live in `easybfe/docking/` (`vina.py`, `embed.py`) and are used by `easybfe ligand
  cdock` (constrained/reference-guided docking) ahead of parameterization.
- `easybfe/protein_prep/` wraps `pdbfixer` for `easybfe protein prep`.
- `easybfe/gbsa/` provides MM/GBSA-style end-state free energy estimation as an alternative/companion to
  full alchemical FEP (`amber.py`, `openmm.py` backends).
- **Analysis that has MD after it must run out of process.** `pymbar` 4 solves through JAX, which takes
  ~75% of the first visible GPU on its first computation and never returns it — dropping every
  reference and forcing a GC frees nothing, because the pool belongs to the XLA backend, which lives
  as long as the process. Measured on an A100 (40 GB): 30781 MiB held, unchanged after `del` + `gc` and
  after idling (`XLA_PYTHON_CLIENT_PREALLOCATE=false` reduces it to 1027 MiB and
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` to a 449 MiB residual context, but neither reaches zero).
  Only one call site has MD still to come: the pre-production estimate in
  `run_abfe_with_early_stop()` (early stop is off by default — `early_stop_threshold` is `None`).
  In-process it starved every later `pmemd` rank with `cudaMalloc Failed out of memory`, which looks
  like an MPS fault and is not one. It therefore goes through `ABFE._analyze_out_of_process()`, which
  runs `analyze_abfe` in a child interpreter and reads back `result.json`; the OS reclaims the GPU when
  the child exits. Every other analysis is the last thing the process does, so it stays in-process.
  See `tests/test_analyze_out_of_process.py`. When diagnosing a GPU OOM in a leg, check
  `nvidia-smi --query-compute-apps=pid,used_memory` for the easybfe process itself before suspecting MPS.
- The retained per-lambda debug scripts (`lambda*/<stage>/<stage>.sh`) invoke serial `pmemd.cuda`, so
  they work for `01.em`–`04.pre_prod` but **not** for the REMD production stage: `05.prod.in` carries
  `numexchg`, which pmemd rejects outside a parallel build. Debug one production window by hand via the
  leg-level groupfile invocation instead.
