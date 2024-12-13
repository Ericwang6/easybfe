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
conda env create -f environment.yml
```

Or you can install the following dependencies with conda and `conda-forge` channel:

+ python >=3.10,<3.13
+ setuptools_scm
+ numpy
+ pandas
+ matplotlib
+ scipy
+ rdkit
+ ambertools
+ acpype
+ openmm >= 8.2.0
+ openmmtools
+ openmmforcefields
+ openff-toolkit
+ [alchemlyb](https://github.com/alchemistry/alchemlyb)
+ [lomap2](https://github.com/OpenFreeEnergy/Lomap)
+ [kartograf](https://github.com/OpenFreeEnergy/kartograf)

### Step 2

Run the following command to install EasyBFE:
```bash
conda activate easybfe
cd easybfe
pip install .
```

## Usage

+ [Command Line Interface (CLI)](docs/cli.md)
+ Python API
+ Web GUI

## Developing Notes

If you want to get involved in the development of have any questions, please create an issuse oe contact Yingze (Eric) Wang: ericwangyz@berkeley.edu