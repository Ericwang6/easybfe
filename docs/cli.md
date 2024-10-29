# Command Line Interface

In this tutorial, we will use the command line interface of EasyBFE to setup RBFE simulations. We will use TYK2 system as an example.

## Step 1: Create a project folder

Easybfe uses "project" to manage a relative binding free energy task. A "project" is located in a specified directory and contains all files involved in a RBFE calculation, including ligands, proteins, simulation files for each perturbation.

To create a RBFE project, use:
```bash
easybfe -d /path/to/project init
```
Replace the `/path/to/project` with the actual path that you want your project to be located.

Then, the project directory will be created. Change your directory to this project directory (the following commands need to be executed in the project directory):
```bash
cd /path/to/project
```

and there will be the following sub-directories:

```bash
-- /path/to/project/
   |-- ligands/
   |-- proteins/
   |-- perturbations/
   |-- uploads/
```
Description of each sub-folder:
+ `ligands`: contains all ligand files, including structure file (.sdf), force field topology file (prmtop)
+ `proteins`: contains all protein structure files (.pdb)
+ `perturbations`: contains all alchemical RBFE simulation files
+ `uploads`: used in Web GUI, CLI users can ignore it

## Step 2: Add a protein

Use this command to add a protein PDB file.

```bash
easybfe add_protein -i examples/tyk2_pdbfixer.pdb -n tyk2
```
Here, users needs to add a **"prepared"** protein PDB file, which means that being hetero atoms deleted, missing atoms/residues added, terminal properly treated, and hydrogens added. It is recommended to use [PDBFixer](https://github.com/openmm/pdbfixer) to prepare a protein file. EasyBFE will check if the added protein PDB file is able to be parametrized with Amber14SB force field with OpenMM, and if you want to disable this check, use `--no-check-ff` flag.

Then you will see in the proteins directory that a protein with name specified in `-n` option has been added:

```
-- /path/to/project/
   |-- proteins/
       |-- tyk2
           |-- tyk2.pdb
```

*New features to add: Wrap PDBFixer in easybfe and offer API*

## Step 3: Add ligands

Use this following command to add ligands and parametrize them:
```bash
easybfe add_ligand -p tyk2 -i examples/tyk2_ligands.sdf -f gaff2 -c bcc -m 10
```
Breakdown of the options:
+ `-p`: specify the name of the protein structure that the ligands belongs to.
+ `-i`: the input ligand structures (.sdf)
+ `-f`: the forcefield to parametrize the ligand. 
+ `-m`: number of processors to run in parallel.

Note: You can add experimental values with `dG.expt` (in kcal/mol) or `affinity.expt` (in uM) in the sdf file and `easybfe report` will use report them together with the FEP values. 

## Step 4: Add perturbations
Use this following command to add perturbations:
```bash
easybfe add_perturbation -p tyk2 -l examples/perturbations.txt --config examples/config_5ns.json -m 10
```
Breakdown of the options:
+ `-p`: specify the name of the protein structure that the ligands belongs to.
+ `-l`: File contains list of perturabtions. In the file, each line contains two ligand names seperated by a whitespace. Besides using `-l` option to add multiple perturbations at the same time, users can also use `--ligandA`, `--ligandB` and `-n` options to add one single perturbation. Use `-h` to see more information.
+ `--config`: Path to the configuration file
+ `-m`: number of processors to run in parallel 

This command will prepare all simulation files (including atom mapping, building dual topology, setting up simulation box, adding solvents/ions) and write a submission file to `perturbations/*/{gas,solvent,complex}/run.slurm` under each perturbation folder. Users need to submit them manually after checking that the atom mapping is good. If you are confident with the atom mapping, you can toogle `--submit` in the command and the files will be submitted automatically. 

## Step 5: Analyze and report

```bash
easybfe analyze -m 10
```

```bash
easybfe report -o report/
```
