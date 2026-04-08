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

## Developing Notes

If you want to contribute to development or have questions, please open an issue or contact Yingze (Eric) Wang: ericwangyz@berkeley.edu
