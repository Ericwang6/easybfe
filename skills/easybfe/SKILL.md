---
name: easybfe-cli
description: Set up and analyze ABFE (absolute) and RBFE (relative) binding free energy calculations using the easybfe CLI, including ligand parameterization. Use when the user asks about running easybfe commands, parameterizing ligands, writing FEP config files, setting up alchemical simulations, or analyzing free energy results.
---

# EasyBFE CLI — ABFE & RBFE Workflows

## Background

**ABFE (Absolute Binding Free Energy)**: computes the absolute binding affinity of a single ligand to a protein. The thermodynamic cycle has three legs — *solvent* (ligand decoupled in water), *complex* (ligand decoupled in the protein binding site), and *restraint* (Boresch orientational restraints turned on/off in the complex). The binding free energy is: ΔG_bind = ΔG_complex − ΔG_solvent + ΔG_restraint.

**RBFE (Relative Binding Free Energy)**: computes the *difference* in binding affinity between two ligands (A → B) via an alchemical mutation. The thermodynamic cycle has two legs — *solvent* (A→B in water) and *complex* (A→B in protein). The relative free energy is: ΔΔG = ΔG_complex − ΔG_solvent.

Both workflows use Amber/pmemd for alchemical MD with Hamiltonian replica-exchange (H-REMD) across lambda windows, and MBAR for free energy estimation.

## CLI Entry Point

```
easybfe [--version] COMMAND [SUBCOMMAND] [OPTIONS]
```

Top-level command groups: `abfe`, `rbfe`, `ligand`, `protein`, `md`.

You can run helpers to have a better and detailed understanding of the functionalities:

```
easybfe COMMAND --help
```

or

```
easybfe COMMAND SUBCOMMAND --help
```

## End-to-End Workflow

Both ABFE and RBFE require parameterized ligand directories as input. A typical pipeline:

```bash
# 1. Parameterize ligands (produces one directory per ligand under -O)
easybfe ligand pargen ligands.sdf -f gaff2 -c gas -O ./ligands

# 2. Set up FEP simulations
easybfe rbfe setup config_rbfe.yaml          # or: easybfe abfe setup config_abfe.yaml

# 3. (Run MD externally — pmemd.cuda / pmemd.cuda.MPI)

# 4. Analyze results
easybfe rbfe analyze ./rbfe/ejm_44~ejm_31    # or: easybfe abfe analyze ./abfe/jmc_23
```

---

## Ligand Parameterization

### `easybfe ligand pargen`

Generate force field parameters (topology + coordinates) for one or more ligands. This is a **prerequisite** for both ABFE and RBFE setup — the setup commands expect parameterized ligand directories containing Amber topology/coordinate files.

```
easybfe ligand pargen LIGAND_FILES... [OPTIONS]
```

The `LIGAND_FILES` accepts SDF, MOL, MOL2, CSV, SMI files. If the input file does not contain 3D information or without explicit hydrogens, the program will add hydrogens and generate 3D structure with rdkit. It can also acce Multi-molecule files produce one output directory per molecule.

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | — | Output directory (single ligand; files written directly here) |
| `--output-base` | `-O` | — | Base dir for per-ligand subdirectories (required for multiple ligands) |
| `--forcefield` | `-f` | `gaff2` | Force field (e.g. `gaff2`, `openff-2.1.0`, or path to `.xml`) |
| `--charge-method` | `-c` | `bcc` | Charge method (`bcc`, `gas`, `resp`) |
| `--engine` | | auto | Backend: `acpype`, `openff`, or custom (auto-detected from FF) |
| `--resp-engine` | | `qchem` | Engine for RESP charges (only when charge-method starts with `resp`) |
| `--nprocs` | `-n` | `-1` | Parallel processes (`-1` = all CPUs, `1` = sequential) |
| `--keep-cache` | | off | Keep intermediate `.smff.tmp` working directory |
| `--raise-errors` | | off | Raise on errors instead of logging and skipping |
| `--no-name-from-stem` | | off | Use SDF/CSV name property instead of filename stem |
| `--only-first` | | off | Only read the first molecule from each file |
| `--name-prop` | | `_Name` | RDKit property for molecule name in SDF |
| `--name-col` | | — | Column name for ligand names (CSV) |
| `--smi-col` | | `smiles` | Column name for SMILES (CSV) |

At least one of `--output` or `--output-base` must be provided.

**Batch from SDF (most common):**

```bash
easybfe ligand pargen ligands.sdf -f gaff2 -O ./ligands
```

**Single ligand:**

```bash
easybfe ligand pargen mol.sdf -f gaff2 -o ./ligands/mol
```

**Multiple input files:**

```bash
easybfe ligand pargen a.sdf b.sdf c.sdf -f gaff2 -O ./ligands
```

**Use different force field**

```bash
easybfe ligand pargen ligands.sdf -f openff-2.1.0 -O ./ligands
```

**Use different charge model**

The program will use `bcc` as the default charge model. Always use `bcc` unless the user explicitly instructed you to use `gas` or `resp`.

```bash
easybfe ligand pargen ligands.sdf -c resp -O ./ligands
```

### Constrained docking reference

For constrained pose generation with `easybfe ligand cdock` (reference-guided embedding, Vina optimization, and optional OpenMM minimization), see:

- [reference/constrained_docking.md](reference/constrained_docking.md)

---

## ABFE Commands

### `easybfe abfe setup`

Prepare ABFE simulation input files from a YAML/JSON config. Ligand inputs must be parameterized directories (output of `ligand pargen`).

```
easybfe abfe setup CONFIG [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--ligand` | `-l` | Override: single ligand directory |
| `--protein` | `-p` | Override: protein PDB path |
| `--output` | `-o` | Override: output directory |
| `--ligand-base` | `-I` | Override: parent directory for ligand paths |
| `--ligand-batch` | `-L` | Override: file listing ligand dirs (one per line) |
| `--output-base` | `-O` | Override: base output dir (batch mode) |
| `--nprocs` | `-n` | Parallel processes (default: auto) |

`--ligand` and `--ligand-batch` are mutually exclusive. Config schema: `AmberAbfeConfig`.

**Basic usage**

```bash
easybfe abfe setup config.yaml
```

**Single ligand:**

```bash
easybfe abfe setup config.yaml -l ligands/jmc_23 -p protein.pdb -o output/jmc_23
```

**Batch (multiple ligands defined in config or via `--ligand-batch`):**

```bash
easybfe abfe setup config.yaml -O ./abfe_output
```

**Config file spec:**
- You should use [assets/config_abfe_5ns.yaml](assets/config_abfe_5ns.yaml) as a template to build up the configuration file and modify the file according to user's specifications. Refer to [reference/abfe_setup_spec.md](reference/abfe_setup_spec.md) for full field reference and modification examples.

**Validate the setup**

After `abfe setup` finishes, confirm each ABFE run directory (single ligand: `output_dir`; batch: each subdirectory under `output_base`) is usable:

- **Leg directories** — `solvent/`, `complex/`, and `restraint/` must all exist.
- **Boresch file** — `boresch.dat` must exist at the same level as those three directories (ABFE run root).
- **Launch scripts** — each of `solvent/`, `complex/`, and `restraint/` must contain a `run.sh` file.

**Structured summary after setup**

When reporting back to the user, include a concise, structured summary:

- **Ligands** — How many distinct ligands are set up for ABFE (single vs batch)?
- **Ligand parameterization** — Ligand force field and charge model from `ligand pargen` (typically `-f` and `-c`); ABFE setup assumes these are already baked into each parameterized ligand directory.
- **Protein and solvent** — Protein force field (`protein_ff`) and water model (`water_ff` / `water_model`) from the setup config (usually the same across `solvent`, `complex`, and `restraint` blocks).
- **Output layout** — Absolute or workspace-relative paths for: (1) parameterized ligand directories, (2) prepared ABFE run directory or `output_base` with per-ligand subdirs.
- **Tree** — A short ASCII tree (a few levels) showing `ligands/` (or equivalent) and `abfe/` (or `output_base`/`output_dir`) with `solvent/`, `complex/`, `restraint/`, and `boresch.dat`.

### `easybfe abfe analyze`

Run MBAR analysis on a completed ABFE simulation directory.

```
easybfe abfe analyze DIRECTORY [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prod-prefix` | `-p` | `05.prod` | Production stage subdirectory name |
| `--temperature` | `-t` | `298.15` | Temperature in Kelvin |
| `--force` | `-f` | off | Re-run MBAR even if `result.json` exists |

The directory must contain `complex/`, `solvent/`, `restraint/`, and `boresch.dat`.

```bash
easybfe abfe analyze ./abfe/jmc_23
easybfe abfe analyze ./abfe/jmc_23 -f   # force re-analysis
```

---

## RBFE Commands

### `easybfe rbfe setup`

Prepare RBFE simulation input files from a YAML/JSON config. Ligand inputs must be parameterized directories (output of `ligand pargen`).

```
easybfe rbfe setup CONFIG [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--protein` | `-p` | Override: protein PDB path |
| `--ligandA` | `-a` | Override: ligand A directory/SDF |
| `--ligandB` | `-b` | Override: ligand B directory/SDF |
| `--ligand-base` | `-I` | Override: parent directory for ligand paths |
| `--ligand-list` | `-L` | Override: text file of ligand paths (one path per line, `#` comments) |
| `--output` | `-o` | Override: output directory (single pair) |
| `--output-base` | `-O` | Override: base output dir (network mode writes `{A}~{B}/` per edge) |

Config schema: `AmberLigandRbfeConfig`.

**Config file spec:**
- You should use [assets/config_rbfe_5ns.yaml](assets/config_rbfe_5ns.yaml) as a template to build up the configuration file and modify the file according to user's specifications. Refer to [reference/rbfe_setup_spec.md](reference/rbfe_setup_spec.md) for full field reference and modification examples.

**Single pair:**

```bash
easybfe rbfe setup config.yaml -a ligands/ejm_44 -b ligands/ejm_31 -p protein.pdb -o output/ejm_44~ejm_31
```

**Network mode (`ligand_list` + `network`):**

```bash
easybfe rbfe setup config.yaml -O ./rbfe_output
```

**Validate the setup**

After `rbfe setup` finishes, confirm each perturbation directory (single pair: `output_dir`; batch: each `{ligandA}~{ligandB}/` under `output_base`) is usable:

- **Required legs** — `solvent/` and `complex/` must exist.
- **Gas leg** — If and only if the config defines a `gas` key (gas-phase leg), `gas/` must also exist; do not expect `gas/` when `gas` was not set.
- **Launch scripts** — every present leg directory (`solvent/`, `complex/`, and `gas/` if applicable) must contain a `run.sh` file.

**Structured summary after setup**

When reporting back to the user, include a concise, structured summary:

- **Ligands** — How many parameterized ligand directories are involved? List or count unique ligand names if helpful.
- **Ligand parameterization** — Ligand force field and charge model from `ligand pargen` (typically `-f` and `-c`) for the directories under `ligand_base`.
- **Protein and solvent** — Protein force field (`protein_ff`) and water model (`water_ff` / `water_model`) from the RBFE config (`solvent` / `complex` / optional `gas` blocks).
- **Perturbations** — How many RBFE perturbations (pairs) were set up? For batch workflows, state whether every ligand that should be in the study appears in at least one pair (full coverage vs partial / user-specified subgraph).
- **Output layout** — Paths for: (1) parameterized ligand parent directory (`ligand_base`, usually the same path given to `ligand pargen` as `--output-base` / `-O`), (2) RBFE output directory (`output_dir` or `output_base` with one subdir per pair).
- **Tree** — A short ASCII tree showing ligand dirs and RBFE output with `solvent/`, `complex/`, and `gas/` only when the gas leg was configured.
- **Network algorithms** — See [reference/rbfe_network_algorithms.md](reference/rbfe_network_algorithms.md) for example `network` blocks.

### `easybfe rbfe analyze`

Run MBAR analysis on a completed RBFE simulation directory.

```
easybfe rbfe analyze DIRECTORY [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prod-prefix` | `-p` | `05.prod` | Production stage subdirectory name |
| `--temperature` | `-t` | `298.15` | Temperature in Kelvin |
| `--force` | `-f` | off | Re-run MBAR even if `result.json` exists |

The directory must contain `complex/` and `solvent/` (optionally `gas/`).

```bash
easybfe rbfe analyze ./rbfe/ejm_44~ejm_31
```

