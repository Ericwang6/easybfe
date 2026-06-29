# EasyBFE

EasyBFE is an open-source software for preparing relative binding free energy calculations for protein-ligand systems.

## Installation

### Step 0

EasyBFE currently supports preparing simulations for AMBER 22 and newer releases that include ACES enhanced sampling and the new soft-core potentials. Install GPU-accelerated AMBER from the official distribution before proceeding.

### Step 1

Clone the repository:

```bash
git clone https://github.com/Ericwang6/easybfe.git
```

Create the conda environment from `environment.yml`:

```bash
cd easybfe
conda env create -f environment.yml -n "easybfe"
```

### Step 2

Install EasyBFE in editable mode:

```bash
conda activate easybfe
cd easybfe
pip install -e .
```

## Example usage

### RBFE
```bash
# Optional: prepare and fix the protein structure
easybfe protein prep tyk2_protein.pdb -o tyk2_protein_fixed.pdb
# Optional: constrained docking to generate poses
easybfe ligand cdock ligands.smi -p tyk2_protein_fixed.pdb -r reference.sdf -O ./ligands
# Parameterize ligands
easybfe ligand pargen ./ligands/*.sdf -f gaff2 -c bcc -O ./ligands
# Edit config.yaml (example: examples/rbfe/config_rbfe_5ns.yaml)
...
# Set up the RBFE workflow
easybfe rbfe setup config.yaml -O ./rbfe
# Submit FEP jobs on your HPC system (set account, partition, and GPU resources as needed)
# Example: loop over lambda windows and legs
for dir in rbfe/*~*/{complex,solvent}
do 
  cd $dir
  sbatch run.sh -A ... -p ... --gres=gpu:A100:4 ...
  cd -
done
# Analyze each perturbation directory
for dir in rbfe/*
do
  easybfe rbfe analyze $dir
done
# Optional: aggregate ΔG across the run tree
easybfe rbfe analyze rbfe/ --dg
```

### ABFE
```bash
# Parameterize the ligand (writes ./ligands/<LIG>/ for each input)
easybfe ligand pargen "/path/to/ligand/<LIG>.sdf" -f gaff2 -c bcc -O ./ligands

# Edit config.yaml (example: examples/abfe/config_abfe_5ns.yaml)
...

# Set up the ABFE workflow
easybfe abfe setup config.yaml -O ./abfe
# Submit FEP jobs (pmemd.cuda / mpirun must be available on PATH)
# Example: loop over each ligand's complex, solvent, and restraint legs
for dir in abfe/*/{complex,solvent,restraint}
do 
  cd $dir
  # Each successful leg leaves a done.tag under complex/, solvent/, or restraint/
  # run.sh is a plain shell script: pass Slurm options via sbatch flags or a #SBATCH header in a wrapper script
  sbatch run.sh -A ... -p ... --gres=gpu:A100:4 ...
  cd -
done

# Analyze each ligand run
for dir in abfe/*
do
  cd $dir
  easybfe abfe analyze .
  cd ..
done
```

### ABFE (one-line pipeline)

The separate `setup` / submit / `analyze` workflow above gives you full control:
each leg (`solvent`, `complex`, `restraint`) is an independent directory, so you
can distribute them across different nodes for additional parallelization. When
you instead want a fully automated, hands-off run for a single protein-ligand
system, use `easybfe abfe pipeline`, which chains every step together —
parameterization, Boresch restraint search, leg setup, running, and analysis —
behind one command (example: `examples/abfe-pipeline/`).

```bash
# Go from a fixed protein PDB + a raw ligand SDF straight to result.json
easybfe abfe pipeline config_5ns.yaml -p ./9qdz_fixed_dry.pdb -l ./1508.sdf -o ./run
```

This single command writes everything under `<ABFE-DIR>` (here `./run`):

```
run/ligand/       parameterized ligand
run/boresch-md/   plain protein-ligand MD + representative structure
run/abfe/         solvent/ complex/ restraint/ + boresch.dat + result.json
run/abfe.log      master log
```

Things the pipeline highlights (see `examples/abfe-pipeline/config_5ns.yaml`):

- **MD-based Boresch restraints.** Setting

```yaml
boresch:
  algorithm: rxrx-md
```

  runs a short plain protein-ligand MD (configured under `boresch_md:`) and
  selects the Boresch anchor atoms and a representative frame from the
  trajectory, rather than picking restraints from a single static pose.

- **Early stopping after equilibration.** The pipeline runs the pre-production
  stages first and estimates the ABFE from the second-to-last stage
  (`04.pre_prod`). If that estimate is weaker than `early_stop_threshold`
  (kcal/mol), the ligand is treated as a weak/non-binder and the expensive final
  production stage (`05.prod`) is skipped to save compute:

```yaml
# Skip 05.prod when the 04.pre_prod estimate is greater than -8.0 kcal/mol.
early_stop_threshold: -8.0
```

- **One line from PDB + SDF to result.** The `pipeline` command takes a fixed
  protein PDB (`-p`), a raw ligand SDF (`-l`), and a config, and produces
  `run/abfe/result.json` with the decomposed and total ΔG, for example:

```json
{
    "complex": 28.46,
    "solvent": 16.23,
    "restraint": -3.90,
    "boresch": 7.61,
    "total": -8.53,
    "total_std": 0.15
}
```

On an HPC system you typically wrap this in a single Slurm script (see
`examples/abfe-pipeline/abfe.slurm`) and submit one job per ligand. Reach for the
separate setup/submit/analyze workflow above when you need to spread the
`solvent`, `complex`, and `restraint` legs across multiple nodes; reach for
`pipeline` when you want the whole thing automated end to end in one place.

## Developing Notes

If you want to contribute to development or have questions, please open an issue or contact Yingze (Eric) Wang: ericwangyz@berkeley.edu
