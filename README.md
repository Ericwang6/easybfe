# EasyBFE

EasyBFE is an open-source software for preparing relative binding free energy calculations for protein-ligand systems.

## Installation

### Step 0:

EasyBFE now only supports preparing simulation for AMBER22 and newer version that supports ACES enhanced sampling and new soft core potentials. Please refer to the AMBER official website to install GPU-accelerated AMBER.

### Step 1:

Clone the repoistory

```bash
git clone https://github.com/Ericwang6/easybfe.git
```

Use the `environment.yml` to install the dependencies:

```bash
cd easybfe
conda env create -f environment.yml -n "easybfe"
```

### Step 2

Run the following command to install EasyBFE:
```bash
conda activate easybfe
cd easybfe
pip install -e .
```

## Example usage

```bash
# Prepare protein, if necessary
easybfe protein prep tyk2_protein.pdb -o tyk2_protein_fixed.pdb
# Run constrained docking to get poses, if necessary
easybfe ligand cdock ligands.smi -p tyk2_protein_fixed.pdb -r reference.sdf -O ./ligands
# Paramterize the ligand
easybfe ligand pargen ./ligands/*.sdf -f gaff2 -c bcc -O ./ligands
# Prepare config.yaml
...
# Setup simulation
easybfe rbfe setup config.yaml -O ./rbfe
# Run FEP simulation with HPC enviornment
# Example:
for dir in rbfe/*~*/{complex,solvent}
do 
  cd $dir
  sbatch run.sh -A ... -p ... --gres=gpu:A100:4 ...
  cd -
done
# Analyze
for dir in rbfe/*
do
  easybfe rbfe analyze $dir
done
easybfe rbfe analyze rbfe/ --dg
```

## Developing Notes

If you want to get involved in the development of have any questions, please create an issuse oe contact Yingze (Eric) Wang: ericwangyz@berkeley.edu