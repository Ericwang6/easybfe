'''
Author: Eric Wang
Date: 10/07/2024

This file contains GAFF-based small molecule force field parameterizer
'''
import warnings
import os, shutil
from pathlib import Path
from typing import List, Union, Optional
import parmed
from rdkit import Chem

from .base import SmallMoleculeForceField
from ..cmd import find_executable, run_command, set_directory


def run_acpype(input: Optional[os.PathLike] = None,
               basename: str = "MOL",
               charge_method: str = "bcc",
               atom_type: str = "gaff2",
               net_charge: Union[int, str] = "auto",
               args: Union[None, List[str]] = None):
    """
    Run acpype

    Parameters
    ----------
    input : str or Path or None
        input file name with extension that `acpype -i` support
    basename : str
        a basename for the project, `acpype -b` option
    charge method : str
        gas, bcc (default), user (user's charges in mol2 file)
    atom_type : str
        atom type, can be 'gaff', 'gaff2', 'amber' (AMBER14SB) or 'amber2' (AMBER14SB + GAFF2), default is gaff2
    net_charge : int or "guess"
        net molecular charge, default is "auto". If "auto" and input is mol/sdf/mol2, iMiner will compute the net charge
        based on input file using RDKit. If "guess", acpype will guess a charge.
    args : List[str] or None
        arguments used to run acpype. if `args` is not None, all other arguments are ignored
    """
    acpype = find_executable("acpype")
    if args is not None:
        cmd = [acpype] + args
    else:
        assert input is not None, "Input is None."
        cmd = [acpype, "-i", str(input), "-b", basename, "-c", charge_method, "-a", atom_type]            
        if net_charge == "auto":
            suffix = Path(input).suffix
            if suffix == ".mol":
                mol = Chem.MolFromMolFile(input, removeHs=False)
            elif suffix == ".sdf":
                mol = Chem.SDMolSupplier(input, removeHs=False)[0]
            elif suffix == ".mol2":
                mol = Chem.MolFromMol2File(input, removeHs=False)
            else:
                mol = None
            if mol:
                net_charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])
            else:
                warnings.warn(f"Fail to parse input file {Path(input).resolve()}. iMiner will let acpype to determine net charge")
        if not isinstance(net_charge, str):
            cmd.extend(["-n", str(net_charge)])
    return_code, out, err = run_command(cmd, raise_error=True)
    return 


class GAFF(SmallMoleculeForceField):
    def __init__(self, atype: str = 'gaff2', charge_method: str = 'bcc'):
        self.atype = atype
        self.charge_method = charge_method
        assert self.atype in ['gaff', 'gaff2'], f'Unsupported atom type: {atype}'
        assert self.charge_method in ['bcc', 'gas'], f'Unsupported charge method: {atype}'
    
    def parametrize(self, ligand_file: os.PathLike, wdir: os.PathLike | None = None):
        ligand_file = Path(ligand_file).resolve()
        assert ligand_file.suffix == '.sdf'
        wdir = Path(wdir).resolve()
        with set_directory(wdir):
            if os.path.isdir('MOL.acpype'):
                shutil.rmtree('MOL.acpype')
            run_acpype(
                ligand_file,
                basename='MOL',
                charge_method=self.charge_method,
                atom_type=self.atype,
                net_charge='auto'
            )
            # amber format
            shutil.copyfile('MOL.acpype/MOL_AC.prmtop', ligand_file.stem + '.prmtop')
            struct = parmed.load_file('MOL.acpype/MOL_AC.prmtop')
            struct.coordinates = Chem.SDMolSupplier(str(ligand_file), removeHs=False)[0].GetConformer().GetPositions()
            struct.save(ligand_file.stem + '.inpcrd', overwrite=True)
            # gmx format
            fp = open(ligand_file.stem + '.top', 'w')
            with open('MOL.acpype/MOL_GMX.itp') as f:
                itp = f.read()
            with open('MOL.acpype/MOL_GMX.top') as f:
                for line in f:
                    if line.startswith('#include "MOL_GMX.itp"'):
                        fp.write(itp)
                        fp.write('\n')
                    elif line.startswith('#'):
                        continue
                    else:
                        fp.write(line)
            fp.close()
