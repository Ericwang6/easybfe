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
+ `-f`: the forcefield to parametrize the ligand. Supported values: `gaff`, `gaff2`, openff series supported by `openff-toolkit` (e.g. `openff-2.1.0`)
+ `-c`: method to assign atomic partial charges. Supported values: `bcc` (AM1-BCC), `gas` (Gasteiger)
+ `-m`: number of processors to run in parallel.

Use `easybfe add_ligand -h` for more information.

**Note**

1. The `-i` option supports the following types of input:

   + One sdf file contains multiple ligands. In this case, the name of each ligand specified in the sdf file (the first line of each mol block) will be used as the name in the project.
   + One sdf file contains one ligand. In this case, by default, the name of this ligand will be the basename of the file. For example, `jmc_23` for `examples/jmc_23.sdf`. Users can also specify its name with `-n` option in this case:
   ```bash
   easybfe add_ligand -p tyk2 -i example/jmc_23.sdf -f gaff2 -c bcc -n jmc_23_custom_name
   ```
   + Multiple sdf files with pattern matching but each file must contain one ligand. If not, only the first ligand will be added. In this case, the basename of each sdf file will be used as the ligand's name.
   ```bash
   easybfe add_ligand -p tyk2 -i example/*.sdf -f gaff2 -c bcc -m 10
   ```

2. When `-i` option takes only one sdf file with one ligand, users can also pass in a customized forcefield (topology) for this ligand through `-f` option. This customized forcefield file has to be in Amber .prmtop format or Gromacs .top format. However, the customized force field should not contain terms other than harmonic bond/angles, peroidic torsions, Lennard-Jones, charge-charge interactions. Torsion-torsion coupling (CMAP) term, virtual sites are not supported.

```bash
easybfe add_ligand -p tyk2 -i example/jmc_23.sdf -f example/jmc_23.prmtop
```
3. You can add experimental values with property name `dG.expt` (in kcal/mol) or `affinity.expt` (in uM) in the sdf file and `easybfe report` will use report them together with the FEP values. 
4. Gasteiger charges is ONLY suitable for debugging. It is strongly not recommended in pratical use. 

## Step 4: Add perturbations
This following command will add a perturbation between `jmc_23` and `jmc_30`:
```bash
easybfe add_perturbation -p tyk2 --ligandA jmc_23 --ligandB jmc_30 -n "jmc_23~jmc_30" --config example/config_5ns.json
```
Breakdown of the options:
+ `-p`: specify the name of the protein structure that the ligands belongs to.
+ `--ligandA`, `--ligandB`: name of the two ligands that form this relative pair. The binding free energy of `{ligandA}` minus the binding free energy of `{ligandB}` will be calculated
+ `-n`: name of this perturbation. If not specified, the default name of `{ligandA}~{ligandB}` will be used. 
+ `--config`: Path to the configuration file.

This command will prepare all simulation files (including atom mapping, building dual topology, setting up simulation box, adding solvents/ions) and write a submission file to `perturbations/*/{solvent,complex}/run.slurm`, where `*` is the name of the perturbation. One should submit them manually after checking that the atom mapping is good. 

**Note**:

1. If you are confident with the atom mapping, you can toogle `--submit` in the command and the files will be submitted automatically. 
2. Users can also prepare multiple perturbations with one command:
   ```bash
   easybfe add_perturbation -p tyk2 -l example/perturbations.txt --config example/config_5ns.json -m 10
   ```
   + `-l`: File contains list of perturabtions. In the file, each line contains two ligand names seperated by a whitespace. For example,
   ```
   jmc_23 jmc_30
   jmc_23 ejm_44
   ```
   + `-m`: number of processors to run in parallel 

   In this option, the perturbations will be named as `jmc_23~jmc_30`, `jmc_23~ejm_44` by default.

## Step 5: Analyze
Use the following command to analyze the simulation. This command will automatically found all perturbations that are finished yet not analyzed and then analyze them. 
```bash
easybfe analyze -m 10
```
Here `-m` specify the number of processors to run in parallel. The analysis will do the following things:

+ Use MBAR to gives free energy estimation for both solvated/complex legs, then gives the final binding free energy differences ($\Delta\Delta G$)
+ Convergence analysis
+ Phase-space overlap analysis
+ Calculate RMSD for end-states
+ Torsion distribution for end-states 

## Step 6: Report
The following command will report the result of the whole project and dump to directory `./report/`.
```bash
easybfe report -o ./report/
```
Easybfe will collect all calculated binding free energies (RBFE) and use the maximum likelihood estimation (MLE) algorithm to give the calculated absolute binding free energy (ABFE) of each ligand. The ABFEs are shifted to make the average of calculated ABFEs the same as the average experimental values.