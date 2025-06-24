import os, glob, shutil
from pathlib import Path
import json
from typing import Optional
from functools import partial
import multiprocessing as mp
from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd

from ...cmd import init_logger


class SimulationStatus(Enum):
    NOTSTART = 0
    RUNNING = 1
    FINISHED = 2
    ERROR = 3


class BaseAmberRbfeProject:
    """
    Base class for a project to run relative binding free energy with Amber
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
        self.md_dir = self.wdir / 'md'
        self.logger = init_logger()

        if init:
            self.wdir.mkdir(exist_ok=True)
            self.ligands_dir.mkdir(exist_ok=True)
            self.proteins_dir.mkdir(exist_ok=True)
            self.rbfe_dir.mkdir(exist_ok=True)
            self.upload_dir.mkdir(exist_ok=True)
            self.md_dir.mkdir(exist_ok=True)
        else:
            assert all(d.is_dir() for d in [self.wdir, self.ligands_dir, self.proteins_dir, self.rbfe_dir, self.upload_dir]), "Not an existing project directory, please use init=True to create one"
    
    @property
    def proteins(self):
        _list = os.listdir(self.proteins_dir)
        _list.sort()
        return _list
    
    @property
    def ligands(self):
        results = defaultdict(list)
        for dirname in self.ligands_dir.glob('*/*'):
            if not Path.is_dir(dirname) or dirname.name.startswith('.'):
                continue
            results[dirname.parent.name].append(dirname.name)
        return results

    def gather_ligands_info(self):
        """
        Gather Ligands info
        """
        infos = []
        for protein_name in self.ligands:
            for ligand_name in self.ligands[protein_name]:
                info_json = self.ligands_dir / protein_name / ligand_name / 'info.json'
                if not info_json.is_file():
                    self.logger.warning(f"Information for ligand {ligand_name} with protein {protein_name} is not complete.")
                    continue
                with open(info_json) as f:
                    info = json.load(f)
                infos.append(info)
        df = pd.DataFrame(infos)
        # empty
        if df.shape[0] == 0:
            return df
        df = df.sort_values(by=['protein', 'name'])
        return df

    def add_ligand(
        self, 
        mol, 
        name: str | None = None,
        protein_name: str | None = None, 
        parametrize: bool = True, 
        forcefield: str = 'gaff2', 
        charge_method: str = 'bcc', 
        overwrite: bool = False,
        expt: float | None = None,
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
        if expt is not None:
            info['dG.expt'] = expt
        elif mol.HasProp('dG.expt'):
            info['dG.expt'] = mol.GetDoubleProp('dG.expt')
        elif mol.HasProp('affinity.expt'):
            info['dG.expt'] = 298.15 * 8.314 * np.log(mol.GetDoubleProp('affinity.expt') * 1e-6) / 1000 / 4.184
        else:
            info['dG.expt'] = None
         
        if parametrize:
            self.logger.info(f"Parametrizing: {lig_file}")
            ff_name = self.parametrize_ligand(name, protein_name, forcefield, charge_method, overwrite)
            info.update({
                "forcefield": ff_name, 
                "charge_method": charge_method
            })

        with open(lig_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=4)
        self.logger.info(f'Ligand {name} is added to : {lig_file}')
        return True

    def add_ligands(
        self, 
        mols,
        num_workers: str | int = 'auto',
        **kwargs
    ):
        """
        Add multiple ligands
        """
        import tempfile
        from rdkit import Chem

        if num_workers == 'auto':
            num_workers = max(1, mp.cpu_count() - 2)

        if isinstance(mols, list) and all(isinstance(m, Chem.Mol) for m in mols):
            pass
        elif isinstance(mols, list) and all(isinstance(m, str) or isinstance(m, Path) for m in mols):
            mols = [str(m) for m in mols]
        elif os.path.isfile(str(mols)):
            stem = Path(mols).stem
            self.logger.info(f"Reading molecules from {mols}")
            if Path(mols).suffix == '.sdf':
                mols = [m for m in Chem.SDMolSupplier(mols, removeHs=False)]
                if len(mols) == 1:
                    name = stem if kwargs.get('name', None) is None else kwargs['name']
                    mols[0].SetProp('_Name', name)
                else:
                    for i, m in enumerate(mols):
                        assert m.GetProp('_Name'), f'Name in molecule {i} is empty'
        elif isinstance(mols, str):
            mols = list(glob.glob(mols))
            mols.sort()
            self.logger.info("Molecules in these files will be added:\n" + "\n".join(mols))
        else:
            raise TypeError("Invalid input")
        
        assert len(mols) > 0, "No molecules found."
        if isinstance(mols[0], Chem.Mol):
            tmpdir = tempfile.mkdtemp()
            tmps = []
            for m in mols:
                tmpfile = os.path.join(tmpdir, f'{m.GetProp("_Name")}.sdf')
                with Chem.SDWriter(tmpfile) as w:
                    w.write(m)
                tmps.append(tmpfile)
            mols = tmps
        
        if len(mols) == 1:
            # If only one molecule, user can input a name, otherwise default names will be used
            self.add_ligand(mols[0], **kwargs)
        else:
            # Names cannot be the same
            names = [Path(m).stem for m in mols]
            for name in names:
                if names.count(name) > 1:
                    raise RuntimeError(f"Duplicated ligand: {name}")
            kwargs.pop('name')
            # Add ligand
            if num_workers == 1:
                for mol in mols:
                    self.add_ligand(mol, None, **kwargs)
            else:
                # Here multiprocessing will clear all properties in the molecule (idk why), so files are used
                func = partial(self.add_ligand, **kwargs)
                with mp.Pool(num_workers) as pool:
                    pool.map(func, mols)
           
    def parametrize_ligand(self, name: str, protein_name: str, forcefield: str = 'gaff2', charge_method: str = 'bcc', overwrite: bool = False):
        from ...smff import GAFF, OpenFF, CustomForceField, SmallMoleculeForceField

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
    
    def run_plain_md(
        self, 
        protein_name: str, 
        ligand_name: str = "",
        task_name: str = "",
        config: os.PathLike = "",
        ligand_only: bool = False,
        submit: bool = False,
    ):
        """
        Setup a plain MD job
        """
        from ..plain_md import create_system, create_workflow

        assert task_name, "Must provide a task_name"
        wdir = self.md_dir / task_name
        wdir.mkdir()

        if ligand_only:
            protein_pdb = ""
        else:
            protein_pdb = self.proteins_dir / protein_name / f'{protein_name}.pdb'
        
        if ligand_name:
            ligand_prmtop = self.ligands_dir / protein_name / ligand_name / f'{ligand_name}.prmtop'
            ligand_inpcrd = self.ligands_dir / protein_name / ligand_name / f'{ligand_name}.inpcrd'
        else:
            ligand_prmtop = ligand_inpcrd = ""
        
        with open(config) as f:
            config = json.load(f)
        
        if ligand_only:
            task_msg = f'ligand {ligand_name}'
            task_type = 'ligand'
        elif ligand_name:
            task_msg = f'protein {protein_name} with ligand {ligand_name}'
            task_type = 'complex'
        else:
            task_msg = f'protein {protein_name}'
            task_type = 'protein'
        
        config['task_type'] = task_type
        with open(wdir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)

        self.logger.info(f"Setting up plain MD task for {task_msg}")
        create_system(
            protein_pdb,
            ligand_prmtop,
            ligand_inpcrd,
            wdir,
            protein_ff=config.get('protein_ff', 'ff14SB'),
            water_ff=config.get('water_ff', 'tip3p'),
            buffer=config.get('buffer', 20.0),
            ionic_strength=config.get('ionic_strength', 0.15),
            do_hmr=config.get('do_hmr', True),
            do_hmr_water=config.get('do_hmr_water', False)
        )

        prmtop = wdir / 'system.prmtop'
        inpcrd = wdir / 'system.inpcrd'

        wf = create_workflow(
            prmtop,
            inpcrd,
            config['workflow'],
            wdir,
            exec=config.get('exec', 'pmemd.cuda')
        )
        wf.header = '\n'.join(config.get('header', []))
        wf.create()
        if submit:
            wf.submit(platform=config.get('submit_platform', 'slurm'))
            self.logger.info("Job submitted")