import os, sys, glob, shutil
from pathlib import Path
import json
from typing import Union, Optional, Dict, Any, Tuple, List
from functools import partial
import logging
import uuid
import multiprocessing as mp
from collections import defaultdict
from enum import Enum
from ..cmd import run_command, set_directory


def init_logger(logname: Optional[os.PathLike] = None) -> logging.Logger:
    # logging
    logger = logging.getLogger(str(uuid.uuid4()))
    logger.propagate = False
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # file
    if logname is not None:
        handler = logging.FileHandler(str(logname))
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


class SimulationStatus(Enum):
    NOTSTART = 0
    RUNNING = 1
    FINISHED = 2
    ERROR = 3



class AmberRbfeProject:
    """
    Project for a Amber Relative Binding Free Energy Calculation
    """
    def __init__(self, wdir: os.PathLike = '.', init: bool = False):
        """
        Initialize a AmberRbfeProject instance

        Parameters
        ----------
        wdir: os.PathLike
            Working directory of the project
        """
        self.wdir = Path(wdir).resolve()
        self.ligands_dir = self.wdir / 'ligands'
        self.proteins_dir = self.wdir / 'proteins'
        self.rbfe_dir = self.wdir / 'rbfe'
        self.upload_dir = self.wdir / 'upload'
        self.logger = init_logger()

        if init:
            self.wdir.mkdir(exist_ok=True)
            self.ligands_dir.mkdir(exist_ok=True)
            self.proteins_dir.mkdir(exist_ok=True)
            self.rbfe_dir.mkdir(exist_ok=True)
            self.upload_dir.mkdir(exist_ok=True)
        else:
            assert all(d.is_dir() for d in [self.wdir, self.ligands_dir, self.proteins_dir, self.rbfe_dir, self.upload_dir]), "Not an existing project directory, please use init=True to create one"
    
    @property
    def proteins(self):
        _list = os.listdir(self.proteins_dir)
        _list.sort()
        return _list
    
    @property
    def ligands(self):
        _list = os.listdir(self.ligands_dir)
        _list.sort()
        return _list
    
    @property
    def perturbations(self) -> Dict[str, List[str]]:
        results = defaultdict(list)
        for pert_dir in self.rbfe_dir.glob('*/*'):
            if not Path.is_dir(pert_dir) or pert_dir.name.startswith('.'):
                continue
            results[pert_dir.parent.name].append(pert_dir.name)
        return results
            
    def add_ligand(
        self, 
        mol, 
        name: str | None = None,
        protein_name: str | None = None, 
        parametrize: bool = True, 
        forcefield: str = 'gaff2', 
        charge_method: str = 'bcc', 
        overwrite: bool = False,
    ):
        """
        Add ligand
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw, Descriptors, Crippen, Lipinski

        if isinstance(mol, str) or isinstance(mol, Path):
            fpath = str(mol)
            suffix = Path(fpath).suffix
            name = Path(fpath).stem if name is None else name
            assert (protein_name is not None) and (protein_name in self.proteins), f"Not a valid protein: {protein_name}"

            if suffix == ".mol":
                mol = Chem.MolFromMolFile(fpath, removeHs=False)
            elif suffix == ".sdf":
                mol = Chem.SDMolSupplier(fpath, removeHs=False)[0]
            elif suffix == ".mol2":
                mol = Chem.MolFromMol2File(fpath, removeHs=False)
            else:
                raise NotImplementedError(f"Unsupported ligand input format: {suffix}")

        if name is None:
            print(mol.GetPropsAsDict())
            name = mol.GetProp('_Name')
            assert name is not None, "Must provide a name"

        mol.SetProp('_Name', name)

        # handle overwrite
        lig_dir = self.ligands_dir / protein_name / name
        if (not overwrite) and lig_dir.is_dir():
            msg = f"Ligand already exists: {lig_dir}"
            self.logger.error(msg)
            raise RuntimeError(msg)
        
        lig_dir.mkdir(parents=True, exist_ok=overwrite)
        lig_file = str(lig_dir / f'{name}.sdf')
        with Chem.SDWriter(lig_file) as w:
            w.write(mol)
        
        # Plot
        mol_noh = Chem.RemoveHs(mol)
        AllChem.Compute2DCoords(mol_noh)
        Draw.MolToFile(mol_noh, lig_dir / f'{name}.png', legend=name, size=(500, 500))

        # Compute properties
        info = dict(
            name = name,
            protein = protein_name,
            mol_weight = Descriptors.MolWt(mol),
            logp = Crippen.MolLogP(mol),
            tpsa = Descriptors.TPSA(mol),
            num_rotatable_bonds = Lipinski.NumRotatableBonds(mol),
            num_h_donors = Lipinski.NumHDonors(mol),
            num_h_acceptors = Lipinski.NumHAcceptors(mol)
        )
         
        self.logger.info(f'Ligand {name} is added to : {lig_file}')
        if parametrize:
            ff_name = self.parametrize_ligand(name, protein_name, forcefield, charge_method, overwrite)
            info.update({
                "forcefield": ff_name, 
                "charge_method": charge_method
            })

        with open(lig_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=4)
        return True

    def add_ligands(
        self, 
        sdf: os.PathLike,
        num_workers: int = 1,
        **kwargs
    ):
        """
        Add multiple ligands
        """
        import tempfile
        from rdkit import Chem

        mols = [m for m in Chem.SDMolSupplier(sdf, removeHs=False)]
        if num_workers == 1:
            for mol in mols:
                self.add_ligand(mol, None, **kwargs)
        else:
            # Here multiprocessing will clear all properties in the molecule (idk why)
            tmpdir = tempfile.mkdtemp()
            tmps = []
            for m in mols:
                tmpfile = os.path.join(tmpdir, f'{m.GetProp("_Name")}.sdf')
                with Chem.SDWriter(tmpfile) as w:
                    w.write(m)
                tmps.append(tmpfile)
            func = partial(self.add_ligand, **kwargs)
            with mp.Pool(num_workers) as pool:
                pool.map(func, tmps)
           
    def parametrize_ligand(self, name: str, protein_name: str, forcefield: str = 'gaff2', charge_method: str = 'bcc', overwrite: bool = False):
        from ..smff import GAFF, OpenFF, CustomForceField, SmallMoleculeForceField

        lig_dir = self.ligands_dir / protein_name / name
        
        ff_name = forcefield
        if isinstance(forcefield, SmallMoleculeForceField):
            ff = forcefield
            ff_name = ff.__class__.__name__
        elif os.path.isfile(forcefield):
            ff = CustomForceField(forcefield, overwrite)
            ff_name = 'custom'
        elif forcefield.startswith('gaff'):
            ff = GAFF(forcefield, charge_method)
        elif forcefield.startswith('openff'):
            ff = OpenFF(forcefield, charge_method) 
        else:
            msg = f"Unsupported force field: {forcefield}"
            self.logger.error(msg)
            raise NotImplementedError(msg)

        ff.parametrize(lig_dir / f'{name}.sdf', lig_dir)
        self.logger.info(f"Ligand {name} is parametrized with {forcefield} and charge method {charge_method}")
        return ff_name

    def add_protein(self, fpath: os.PathLike, name: Optional[str] = None, check_ff: bool = True, overwrite: bool = False):
        """
        Add protein to the project

        Parameters
        ----------
        fpath: os.PathLike
            Path to the protein file to be added. Only pdb format is supported
        name: str
            Name of the protein. If None, will be inferred from the input path
        check_ff: bool
            Whether to check the protein can be parametrized by Amber14SB force field.
        overwrite: bool
            Whether overwrite existing protein directory. Default False.
        """
        suffix = Path(fpath).suffix
        name = Path(fpath).stem if name is None else name
        assert suffix == '.pdb', 'Only PDB format is supported'
        prot_dir = self.proteins_dir / name
        
        if (not overwrite) and prot_dir.is_dir():
            msg = f"Protein already exists: {prot_dir}"
            self.logger.error(msg)
            raise RuntimeError(msg)
        
        prot_dir.mkdir(exist_ok=overwrite)
        fprot = prot_dir / f'{name}{suffix}'
        shutil.copyfile(fpath, fprot)
        self.logger.info(f'Protein {name} is added: {fprot}')

        if check_ff:
            self.logger.info("Checking if protein can be parametrized with Amber force field")
            import openmm.app as app
            ff = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
            try:
                ff.createSystem(app.PDBFile(fpath).topology)
                self.logger.info('Force field check Passed.')
            except Exception as e:
                self.logger.error(f'The protein can not be parametrized with Amber force field.')
                raise e
        return True
    
    def add_perturbation(
        self, 
        ligandA_name: str, 
        ligandB_name: str, 
        protein_name: str,
        pert_name: Optional[str] = None, 
        atom_mapping_method: str = 'kartograf',
        atom_mapping_options: Dict[str, Any] = dict(), 
        config: Dict[str, Any] | os.PathLike = dict(),
        submit: bool = False,
        skip_gas: bool = True,
        overwrite: bool = False
    ):
        """
        Create a perturbation pair (relative binding free energy between ligand A and ligand B) 
        and set up FEP simulation

        Parameters
        ----------
        ligandA_name: str
            Name of ligand A
        ligandB_name: str
            Name of ligand B
        protein_name: str
            Name of the protein
        pert_name: str
            Name of this perturbation. If None, will use `ligandA_name`~`ligandB_name`.`protein_name`
        mcs: str or os.PathLike
            Maximum common structure (MCS) between ligand A and B. If a sdf file or a SMARTS string is provided, rdkit will be used 
            to parse it. If None, will use rdkit to determine the MCS. Manually provided MCS is highly recommended.
        config: dict or os.PathLike
            JSON-formatted configuration file or dictionary. Must be provided.
        submit: bool
            Whether to use SLURM (sbatch command) to submit the simulation job. Default False.
        skip_gas: bool
            Whether to skip the gas-phase FEP simulation, only useful when `submit=True`. Default True.
        overwrite: bool
            Whether to overwrite existing perturbation directory. Default False.
        patch_func: Callable
            Function to modify the regular behavior of this function, used for modifying force field parameters
        """
        import numpy as np
        from rdkit import Chem
        import openmm.unit as unit
        from ..mapping import LazyMCSMapper, OpenFEAtomMapper
        from .prep import prep_ligand_rbfe_systems
        from .op import fep_workflow

        # Read config from json file
        if not isinstance(config, dict):
            with open(config) as f:
                config = json.load(f)
        
        # Setup
        if pert_name is None:
            pert_name = f"{ligandA_name}~{ligandB_name}"
        
        pert_dir = self.rbfe_dir / protein_name / pert_name
        if (not overwrite) and pert_dir.is_dir():
            msg = f'Perturbation already exists: {pert_dir}'
            self.logger.error(msg)
            raise RuntimeError(msg)
            
        pert_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f'Creating dirctory: {pert_dir}')

        with open(pert_dir / 'info.json', 'w') as f:
            basic_info = {
                "protein_name": protein_name,
                "ligandA_name": ligandA_name, 
                "ligandB_name": ligandB_name,
            }
            json.dump(basic_info, f, indent=4)
        
        # Read Ligands
        ligandA = Chem.SDMolSupplier(str(self.ligands_dir / protein_name / ligandA_name / f'{ligandA_name}.sdf'), removeHs=False)[0]
        ligandB = Chem.SDMolSupplier(str(self.ligands_dir / protein_name / ligandB_name / f'{ligandB_name}.sdf'), removeHs=False)[0]

        # Atom Mapping
        self.logger.info("Generate atom mapping")
        if atom_mapping_method == 'lazymcs':
            mcs = atom_mapping_options.get('mcs', None)
            if mcs is None:
                self.logger.info("No MCS passed in. Will use RDKit to generate one.")
                pass
            elif isinstance(mcs, Chem.Mol):
                pass
            elif isinstance(mcs, Path) or (isinstance(mcs, str) and os.path.isfile(mcs)):
                self.logger.info(f"Read MCS from {mcs}")
                mcs = Chem.SDMolSupplier(str(mcs), removeHs=True)[0]
            else:
                mcs = Chem.MolFromSmiles(mcs)
                if mcs is None:
                    self.logger.info("MCS is not a valid SMILES. Parsing MCS as a SMARTS.")
                    mcs = Chem.MolFromSmarts(mcs)
                if mcs is None:
                    msg = f"Unrecognized MCS: '{mcs}'. Maybe this is not a valid SMILES or SMARTS or file path"
                    self.logger.error(msg)
                    raise RuntimeError(mcs)
            atom_mapping_options['mcs'] = mcs
            mapper = LazyMCSMapper(**atom_mapping_options)
        else:
            mapper = OpenFEAtomMapper(method=atom_mapping_method, **atom_mapping_options)    
        atom_mapping = mapper.run(ligandA, ligandB, pert_dir)

        tol = 0.1
        for k, v in atom_mapping.items():
            posA = ligandA.GetConformer().GetAtomPosition(k)
            posB = ligandB.GetConformer().GetAtomPosition(v)
            dist = np.linalg.norm(posA - posB)
            if dist >= tol:
                msg = f"Distance ({dist:.2f} angstrom) between mapped atoms Atom(#{k}) in ligand A and Atom(#{v}) in ligand B larger than {tol:.2f} angstrom"
                self.logger.warning(msg)
                ligandB.GetConformer().SetAtomPosition(v, [float(x) for x in posA])
                self.logger.warning(f"Reset Atom(#{v}) position: {(posB.x, posB.y, posB.z)} -> {(posA.x, posA.y, posA.z)}")

        # Prep simulation system
        self.logger.info("Preparing simulation system")
        prep_dir = pert_dir / 'prep'
        prep_dir.mkdir(exist_ok=True)
        prep_ligand_rbfe_systems(
            self.proteins_dir / protein_name / f'{protein_name}.pdb',
            ligandA,
            self.ligands_dir / protein_name / ligandA_name / f'{ligandA_name}.prmtop',
            ligandB,
            self.ligands_dir / protein_name / ligandB_name / f'{ligandB_name}.prmtop',
            mapping=atom_mapping,
            wdir=prep_dir,
            protein_ff=config.get('protein_ff', 'ff14SB'),
            water_ff=config.get('water_ff', 'tip3p'),
            neuturalize=config.get('neuturalize', True),
            ionic_strength=config.get('ionic_strength', 0.15) * unit.molar,
            gas_buffer=config.get('gas', {}).get('buffer', 20.0) / 10 * unit.nanometers,
            solvent_buffer=config.get('solvent', {}).get('buffer', 15.0) / 10 * unit.nanometers,
            complex_buffer=config.get('complex', {}).get('buffer', 12.0) / 10 * unit.nanometers,
        )
        with open(prep_dir / 'mask.json') as f:
            mask = json.load(f)

        # Prep workflow
        self.logger.info("Preparing Amber simulation inputs")
        for leg in ['gas', 'solvent', 'complex']:
            leg_dir = pert_dir / leg
            leg_dir.mkdir(exist_ok=True)
            prmtop = pert_dir / f'prep/{leg}.prmtop'
            inpcrd = pert_dir / f'prep/{leg}.inpcrd'

            leg_config = {
                "inpcrd": str(inpcrd),
                "prmtop": str(prmtop),
            }
            leg_config.update(config[leg])
            leg_config.update(mask)

            with open(leg_dir / 'config.json', 'w') as f:
                json.dump(leg_config, f, indent=4)
        
            fep_workflow(leg_config, leg_dir, gas_phase=(leg == 'gas'))
            self.logger.info(f"FEP simulation workflow is set for leg: {leg}. Config file written to: {leg_dir / 'config.json'}")

            with open(Path(__file__).parent / 'submit.slurm') as f:
                slurm = f.read()
                slurm = slurm.replace('@SLURM_CONFIG', '\n'.join(config['slurm_config']))
                slurm = slurm.replace('@SET_UP_ENV', f'{config["env"]}' if 'env' in config else '')
                slurm = slurm.replace('@NUM_LAMBDA', str(len(config[leg]['lambdas'])))
                slurm = slurm.replace(
                    '@STAGES', 
                    '("em" "heat" "pres_0" "pres_1" "pres_2" "pre_prod")' if leg != 'gas' else '("em" "heat")'
                )
            with open(leg_dir / 'run.slurm', 'w') as f:
                f.write(slurm)
        
        # submit
        if submit:
            legs = ['ligands', 'complex'] if skip_gas else ['ligands', 'complex', 'gas']
            for leg in legs:
                with set_directory(pert_dir / leg):
                    _, out, _ = run_command(['sbatch', 'run.slurm'])
                self.logger.info(f"Job submitted for {leg}: {out.split()[-1]}")
    
    def add_perturbations(
        self,
        perts: List[Tuple[str, str]],
        protein_name: str,
        num_workers: int = 1,
        **kwargs
    ):
        """
        Add Perturbations
        """
        if num_workers == 1:
            for pert in perts:
                self.add_perturbation(pert[0], pert[1], protein_name, **kwargs)
        else:
            func = partial(self.add_perturbation, protein_name=protein_name, **kwargs)
            with mp.Pool(num_workers) as pool:
                pool.starmap(func, perts)

    def analyze_pert(self, protein_name: str, pert_name: str, skip_traj: bool = False):
        self.evaluate_free_energy(protein_name, pert_name)
        if not skip_traj:
            self.process_traj(protein_name, pert_name)
    
    @classmethod
    def query_perturbation_status(cls, pert_dir: Path):
        status = {
            'analysis': Path.is_file(pert_dir / 'result.json')
        }
        legs = ['gas', 'solvent', 'complex']
        for leg in legs:
            if Path.is_file(pert_dir / leg / 'done.tag'):
                status[leg] = SimulationStatus.FINISHED
            elif Path.is_file(pert_dir / leg / 'error.tag'):
                status[leg] = SimulationStatus.ERROR
            elif Path.is_file(pert_dir / leg / 'eq.tag') or Path.is_file(pert_dir / leg / 'prod.tag'):
                status[leg]= SimulationStatus.RUNNING
            else:
                status[leg] = SimulationStatus.NOTSTART
        return status
    
    @classmethod
    def query_perturbation_info(cls, pert_dir: Path):
        with open(pert_dir / 'info.json') as f:
            info = json.load(f)
        info['pert_name'] = pert_dir.name
        status = cls.query_perturbation_status(pert_dir)
        info['analysis.finished'] = status['analysis']
        for leg in ['gas', 'solvent', 'complex']:
            info[f'{leg}.status'] = status[leg].name
        if info['analysis.finished']:
            with open(pert_dir / 'result.json') as f:
                result = json.load(f)
            for key1 in result:
                for key2 in result[key1]:
                    info[f'{key1}.{key2}'] = result[key1][key2]
        return info
    
    def gather_perturbations_info(self):
        '''Gather data of all perturbations'''
        import pandas as pd

        infos = []
        for protein_name, pert_names in self.perturbations.items():
            for pert_name in pert_names:
                info = self.query_perturbation_info(self.rbfe_dir / protein_name / pert_name)
                infos.append(info)
        df = pd.DataFrame(infos)
        return df
    
    def analyze_pert(self, protein_name: str, pert_name: str, skip_traj: bool = False):
        self.evaluate_free_energy(protein_name, pert_name)
        if not skip_traj:
            self.process_traj(protein_name, pert_name)
    
    def analyze(self, interactive=True):
        info_df = self.gather_perturbations_info()
        query_str = '~`analysis.finished` & `solvent.status` == "{}" & `complex.status` == "{}"'.format(
            SimulationStatus.FINISHED.name, SimulationStatus.FINISHED.name
        )
        ana_df = info_df.query(query_str)
        print('Found these perturbations needs analysis:')
        print(ana_df[['protein_name', 'ligandA_name', 'ligandB_name', 'ddG.expt']])
        if interactive:
            start = input('Enter y to start: ') == 'y'
        else:
            start = True
        if start:
            for _, row in ana_df.iterrows():
                self.analyze_pert(row['protein_name'], row['pert_name'])

    def evaluate_free_energy(self, protein_name: str, pert_name: str):
        """
        Analyze FEP simulation results using `alchemlyb` package.
        Free energy will be estimated using MBAR, overlap matrix and convergence analysis are also performed.

        For details, one can refer to: J Comput Aided Mol Des (2015) 29:397-411

        Parameters
        ----------
        pert_name: str
            Name of the perturbation to analyze
        skip_gas: bool
            Whether to skip gas-phase simulation analysis. Default is True.
        """
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        import alchemlyb
        from alchemlyb.estimators import MBAR
        from alchemlyb.parsing.amber import extract_u_nk
        from alchemlyb.convergence import forward_backward_convergence
        from alchemlyb.visualisation.convergence import plot_convergence
        from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix

        dG = {}
        dG_std = {}

        ddG = {}
        ddG_std = {}

        pert_dir = self.rbfe_dir / protein_name / pert_name

        skip_gas = not os.path.isfile(pert_dir / 'gas' / 'done.tag')
        self.logger.info("Found gas-pahse simulation")

        legs = ['solvent', 'complex'] if skip_gas else ['solvent', 'complex', 'gas']
        for leg in legs:
            leg_dir = pert_dir / leg
            with open(leg_dir / 'config.json') as f:
                T = json.load(f).get('temperature', 298.15)
                kBT = 8.314 * T / 1000 / 4.184

            self.logger.info(f"Performing MBAR for {leg}")
            self.logger.info("Extracting data from output...")
            u_nks = []
            num_lambda = len(list(leg_dir.glob('lambda*')))
            for i in tqdm(range(num_lambda), leave=True):
                out = str(leg_dir / f"lambda{i}/prod/prod.out")
                u_nks.append(extract_u_nk(out, T=T))
            
            # evaluate free energy with MBAR
            self.logger.info("Running MBAR estimator...")
            mbarEstimator = MBAR()
            mbarEstimator.fit(alchemlyb.concat(u_nks))
            dG[leg] = mbarEstimator.delta_f_.iloc[0, -1] * kBT
            dG_std[leg] = mbarEstimator.d_delta_f_.iloc[0, -1] * kBT
            
            # convergence analysis
            self.logger.info("Running convergence analysis...")
            conv_df = forward_backward_convergence(u_nks, "mbar")
            for key in ['Forward', 'Forward_Error', 'Backward', 'Backward_Error']:
                conv_df[key] *= kBT
                conv_df.to_csv(leg_dir /"convergence.csv", index=None)
                conv_ax = plot_convergence(conv_df)
                conv_ax.set_ylabel("$\Delta G$ (kcal/mol)")
                conv_ax.set_title(f"Convergence Analysis - {leg.capitalize()}")
                conv_ax.figure.savefig(str(leg_dir /"convergence.png"), dpi=300)

            # overlap matrix
            self.logger.info("Plotting overlap matrix...")
            overlap_ax = plot_mbar_overlap_matrix(mbarEstimator.overlap_matrix)
            overlap_ax.figure.savefig(str(leg_dir /"overlap.png"), dpi=300)

        convergence = {leg: pd.read_csv(str(pert_dir / leg / 'convergence.csv')) for leg in legs}

        pairs = [('complex', 'solvent'), ('solvent', 'gas'), ('complex', 'gas')]
        names = ['total', 'solvation', 'complex']

        for (leg1, leg2), name in zip(pairs, names):
            ddG[name] = dG[leg1] - dG[leg2]
            ddG_std[name] = np.linalg.norm([dG_std[leg1], dG_std[leg2]])

            ddG_conv_df = convergence['complex'].copy()
            for tag in ['Forward', 'Backward']:
                ddG_conv_df[tag] = convergence[leg1][tag] - convergence[leg2][tag]
                ddG_conv_df[f'{tag}_Error'] = np.sqrt(
                    convergence[leg1][f'{tag}_Error'] ** 2 + convergence[leg2][f'{tag}_Error'] ** 2
                )
            ddG_conv_df.to_csv(pert_dir / f"{name}_convergence.csv", index=None)
            conv_ax = plot_convergence(ddG_conv_df)
            conv_ax.set_ylabel("$\Delta\Delta G$ (kcal/mol)")
            conv_ax.set_title(f"Convergence Analysis")
            conv_ax.figure.savefig(str(pert_dir / f"{name}_convergence.png"), dpi=300)

            if skip_gas:
                break

        with open(pert_dir / 'result.json', 'w') as f: 
            json.dump({"dG": dG, "dG_std": dG_std, 'ddG': ddG, 'ddG_std': ddG_std}, f, indent=4)
        
        self.logger.info(f"Finished - free energy evaulation. Results written to {pert_dir / 'result.json'}")

    def process_traj(self, protein_name: str, pert_name: str):
        """
        Processing trajectory of RBFE simulation endpoints: remove unphysical atoms, remove PBC, alignment 
        """
        from tqdm import tqdm
        import MDAnalysis as mda
        from MDAnalysis.analysis import align
        
        perturb_dir = self.rbfe_dir / protein_name / pert_name
        
        for leg in ['solvent', 'complex']:
            num_lambdas = len(glob.glob(os.path.join(perturb_dir, f'{leg}/lambda*/')))

            for lmd, resid in zip([0, num_lambdas - 1], [1, 2]):

                # prmtop = os.path.join(perturb_dir, f'prep/{leg}_solvated.prmtop')
                in_top = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/prod.pdb')
                in_trj = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/prod.mdcrd')

                out_pdb = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/prod_traj.pdb')
                out_trj = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/prod_traj.xtc')
                trj_dir = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/traj')
                if leg == 'complex' and not os.path.isdir(trj_dir):
                    os.mkdir(trj_dir)

                if leg == 'complex':
                    sele_str = f'protein or resid {resid}'
                else:
                    sele_str = f'resid {resid}'

                u = mda.Universe(in_top, in_trj, format='NCDF')
                u_ref = mda.Universe(in_top, in_trj, format='NCDF')

                alignment = align.AlignTraj(u, u_ref, select=sele_str, in_memory=True)
                alignment.run()

                selection = u.select_atoms(sele_str)
                with mda.Writer(out_trj, n_atoms=selection.n_atoms) as W:
                    for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory)):
                        W.write(selection)
                        if leg == 'complex':
                            selection.write(os.path.join(trj_dir, f'traj{i}.pdb'))
                        if i == 0:
                            selection.write(out_pdb)