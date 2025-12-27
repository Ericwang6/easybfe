'''
Author: Eric Wang
Date: 09/11/2025

GBSA calculation in EasyBFE
'''
import os, glob
from pathlib import Path
from typing import Literal, List

from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
import parmed
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader

from ..cmd import run_command
from ..smff.gaff import GAFF


leap_in = '''source leaprc.protein.{protein_ff}
source leaprc.{ligand_ff}
loadoff {ligand_lib}
loadamberparams {ligand_frcmod}
ligand = loadmol2 {ligand_mol2}
protein = loadpdb {protein_pdb}
complex = combine {{ protein ligand }}
set default PBRadii mbondi2
saveamberparm complex complex.prmtop complex.inpcrd
savepdb complex complex.pdb
saveamberparm protein protein.prmtop protein.inpcrd
saveamberparm ligand ligand.prmtop ligand.inpcrd
quit'''


gbsa_in = '''&general
  startframe           = 1                                              
  endframe             = 9999999                                                                                     
/

# (AMBER) Generalized-Born namelist variables
&gb
  igb                  = {igb}                                                                                       
  saltcon              = {saltcon}
  epsin = {epsin}
  epsout = {epsout}
/'''


def prep_parameters(
    protein_pdb: os.PathLike,
    ligand_sdf: os.PathLike,
    wdir: os.PathLike = '.',
    protein_ff: str = 'ff14SB',
    ligand_ff: Literal['gaff', 'gaff2'] = 'gaff2',
    charge_method: Literal['bcc', 'gas'] = 'bcc',
    reuse_cache: bool = False
):

    protein_pdb = Path(protein_pdb).resolve()
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    GAFF(atype=ligand_ff, charge_method=charge_method, reuse_cache=reuse_cache).parametrize(ligand_sdf, wdir=wdir)
    ligand_lib = wdir /  'MOL.acpype/MOL_AC.lib'
    ligand_mol2 = wdir / f'MOL.acpype/MOL_{charge_method}_{ligand_ff}.mol2'
    ligand_frcmod = wdir / 'MOL.acpype/MOL_AC.frcmod'

    with open(wdir / 'leap.in', 'w') as f:
        f.write(leap_in.format(
            protein_ff=protein_ff,
            ligand_ff=ligand_ff,
            ligand_lib=ligand_lib,
            ligand_mol2=ligand_mol2,
            ligand_frcmod=ligand_frcmod,
            protein_pdb=protein_pdb
        ))
    
    run_command('tleap -f leap.in', cwd=wdir)


def prep_gbsa_input(
    save_path: os.PathLike = 'gbsa.in',
    igb: int = 2, saltcon: float = 0.15, epsin: float = 4.0, epsout: float = 80.0,
):
    with open(save_path, 'w') as f:
        f.write(gbsa_in.format(
            igb=igb,
            saltcon=saltcon,
            epsin=epsin,
            epsout=epsout
        ))


def parse_csv(fcsv):
    with open(fcsv) as f:
        header = []
        data = []
        _read = False
        for line in f:
            if line.startswith('DELTA Energy Terms'):
                _read = True
                header = f.readline().strip().split(',')
                continue
            if _read:
                line = line.strip()
                if line == '':
                    break
                content = line.split(',')
                content[0] = int(content[0])
                content[1:] = [float(x) for x in content[1:]]
                data.append(content)
    return pd.DataFrame(data, columns=header)
        

def run_gbsa_with_prmtops(
    protein_prmtop: os.PathLike,
    ligand_prmtop: os.PathLike,
    complex_prmtop: os.PathLike,
    traj_file: os.PathLike,
    gbsa_in: os.PathLike = '',
    wdir: os.PathLike = '.',
    remove_tmp: bool = True,
):
    protein_prmtop = Path(protein_prmtop).resolve()
    ligand_prmtop = Path(ligand_prmtop).resolve()
    complex_prmtop = Path(complex_prmtop).resolve()
    # NOTE: here the trajectory file must be in a format that cpptraj can read
    # and the atom order should be first protein then ligand
    traj_file = Path(traj_file).resolve()
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)

    if os.path.isfile(gbsa_in):
        gbsa_in = Path(gbsa_in).resolve()
    elif Path.is_file(wdir / 'gbsa.in'):
        gbsa_in = wdir / 'gbsa.in'
    else:
        prep_gbsa_input(save_path=wdir / 'gbsa.in')
        gbsa_in = wdir / 'gbsa.in'
        warnings.warn(
            'Neither GBSA input file is provided nor "gbsa.in" is found in the workding directory'
            ' A default one is generated.'
        )

    command = f"MMPBSA.py -O -i {gbsa_in} -cp {complex_prmtop} -rp {protein_prmtop} -lp {ligand_prmtop} -y {traj_file} -eo gbsa_out.csv"
    results = run_command(command, cwd=wdir)

    if remove_tmp:
        for f in glob.glob(os.path.join(wdir, '_MMPBSA*')):
            os.remove(f)
    
    df = parse_csv(wdir / 'gbsa_out.csv')
    df.to_csv(wdir / 'gbsa.csv', index=None)
    
    return df


def run_gbsa(
    protein_pdb: os.PathLike,
    ligand_sdf: os.PathLike,
    traj_file: os.PathLike,
    wdir: os.PathLike = '.',
    protein_ff: str = 'ff14SB',
    ligand_ff: Literal['gaff', 'gaff2'] = 'gaff2',
    charge_method: Literal['bcc', 'gas'] = 'bcc',
    remove_tmp: bool = True,
    igb: int = 2, saltcon: float = 0.15, epsin: float = 4.0, epsout: float = 80.0
):
    
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)

    traj_file = Path(traj_file).resolve()

    prep_parameters(
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        wdir=wdir,
        protein_ff=protein_ff,
        ligand_ff=ligand_ff,
        charge_method=charge_method,
    )

    prep_gbsa_input(
        save_path=wdir / 'gbsa.in',
        igb=igb,
        saltcon=saltcon,
        epsin=epsin,
        epsout=epsout
    )

    return run_gbsa_with_prmtops(
        protein_prmtop=wdir / 'protein.prmtop',
        ligand_prmtop=wdir / 'ligand.prmtop',
        complex_prmtop=wdir / 'complex.prmtop',
        traj_file=traj_file,
        gbsa_in=wdir / 'gbsa.in',
        wdir=wdir,
        remove_tmp=remove_tmp
    )


def run_gbsa_for_ligand_conformers(
    protein_pdb: os.PathLike,
    ligand_sdf: os.PathLike,
    ligand_confs: os.PathLike | List[os.PathLike] | np.ndarray,
    wdir: os.PathLike = '.',
    protein_ff: str = 'ff14SB',
    ligand_ff: Literal['gaff', 'gaff2'] = 'gaff2',
    charge_method: Literal['bcc', 'gas'] = 'bcc',
    remove_tmp: bool = True,
    igb: int = 2, saltcon: float = 0.15, epsin: float = 4.0, epsout: float = 80.0,
    run_em: bool = True,
    em_constraint: bool = True
):

    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)

    prep_parameters(
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        wdir=wdir,
        protein_ff=protein_ff,
        ligand_ff=ligand_ff,
        charge_method=charge_method,
    )

    prep_gbsa_input(
        save_path=wdir / 'gbsa.in',
        igb=igb,
        saltcon=saltcon,
        epsin=epsin,
        epsout=epsout
    )

    ref = Chem.SDMolSupplier(ligand_sdf, removeHs=False)[0]
    ref_key = Chem.MolToInchiKey(ref)
    names = None
    if isinstance(ligand_confs, np.ndarray):
        assert ligand_confs.shape[1] == ref.GetNumAtoms(), f'Number of atoms in the ligand conformers ({traj.shape[1]}) does not match that in the ligand sdf file ({ref.GetNumAtoms()})'
        ligand_positions = ligand_confs
    else:
        if isinstance(ligand_confs, str) or isinstance(ligand_confs, Path):
            mols = [m for m in Chem.SDMolSupplier(str(ligand_confs), removeHs=False)]
        else:
            names = ligand_confs
            mols = [Chem.SDMolSupplier(f, removeHs=False)[0] for f in ligand_confs]
        # for m in mols:
        #     assert Chem.MolToInchiKey(m) == ref_key, 'Molecule in the trajectory does not match that in the ligand sdf file'
        ligand_positions = np.array([m.GetConformer().GetPositions() for m in mols])


    complex_positions = []
    protein_coords = parmed.load_file(str(wdir/'protein.prmtop'), xyz=str(wdir/'protein.inpcrd')).coordinates / 10
    if run_em:
        import openmm.app as app
        import openmm as mm 

        prmtop = app.AmberPrmtopFile(str(wdir / 'complex.prmtop'))
        system = prmtop.createSystem(nonbondedMethod=app.CutoffNonPeriodic)

        if em_constraint:
            for residue in prmtop.topology.residues():
                if residue.name == 'MOL':
                    continue
                for atom in residue.atoms():
                    if atom.element.symbol == 'H':
                        continue
                    system.setParticleMass(atom.index, 0.0)
        
        sim = app.Simulation(
            prmtop.topology,
            system,
            mm.LangevinIntegrator(300, 1.0, 0.001)
        )
        for ligand_position in ligand_positions:
            start_pos = np.vstack((protein_coords, ligand_position / 10))
            sim.context.setPositions(start_pos)
            sim.minimizeEnergy()
            state = sim.context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True)._value * 10
            complex_positions.append(positions)
    else:
        for c in ligand_positions:
            complex_positions.append(np.vstack((protein_coords, c)))

    traj_file = wdir / 'traj.xtc'
    u = mda.Universe(str(wdir / 'complex.prmtop'))
    u.trajectory = MemoryReader(np.array(complex_positions))
    with mda.Writer(str(traj_file), n_atoms=u.atoms.n_atoms) as W:
        for ts in u.trajectory:
            W.write(u.atoms)

    df = run_gbsa_with_prmtops(
        protein_prmtop=wdir / 'protein.prmtop',
        ligand_prmtop=wdir / 'ligand.prmtop',
        complex_prmtop=wdir / 'complex.prmtop',
        traj_file=traj_file,
        gbsa_in=wdir / 'gbsa.in',
        wdir=wdir,
        remove_tmp=remove_tmp
    )
    if names:
        df['Frame #'] = names

    df.to_csv(wdir / 'gbsa.csv', index=None)
    return df
