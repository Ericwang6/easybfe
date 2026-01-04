'''
Author: Eric Wang
Date: 10/07/2024

This file contains GAFF-based small molecule force field parameterizer
'''
import warnings
import os, shutil
from pathlib import Path
from typing import List, Union, Optional, Literal
import parmed
from rdkit import Chem

from .base import SmallMoleculeForceField
from ..cmd import find_executable, run_command, set_directory
from .utils import process_prmtop


def run_acpype(input: Optional[os.PathLike] = None,
               basename: str = "MOL",
               charge_method: str = "bcc",
               atom_type: str = "gaff2",
               net_charge: Union[int, str] = "auto",
               args: Union[None, List[str]] = None):
    """
    Run acpype to generate AMBER and GROMACS force field parameters.
    
    acpype is a tool that automatically generates topology files for AMBER
    and GROMACS from small molecule structures. This function provides a
    Python interface to run acpype with various options.
    
    Parameters
    ----------
    input : os.PathLike, optional
        Input file path with extension that acpype supports (e.g., .mol,
        .sdf, .mol2). Required if `args` is None.
    basename : str, default "MOL"
        Basename for the acpype project. This determines the prefix for
        output files (e.g., "MOL.acpype", "MOL_AC.prmtop").
    charge_method : str, default "bcc"
        Method for assigning partial charges. Options:
        
        * "bcc": AM1-BCC charges (default)
        * "gas": Gasteiger charges
        * "user": Use charges from the input mol2 file
    
    atom_type : str, default "gaff2"
        Atom typing scheme to use. Options:
        
        * "gaff": General Amber Force Field (GAFF)
        * "gaff2": GAFF2 (default)
        * "amber": AMBER14SB protein force field
        * "amber2": AMBER14SB + GAFF2
    
    net_charge : int or str, default "auto"
        Net molecular charge. Options:
        
        * "auto": Automatically compute charge from input file using RDKit
          (only works for .mol, .sdf, .mol2 files)
        * "guess": Let acpype guess the charge
        * int: Explicit charge value
    
    args : list of str, optional
        Custom command-line arguments to pass directly to acpype. If provided,
        all other parameters are ignored and acpype is run with only these
        arguments.
    
    Raises
    ------
    AssertionError
        If `input` is None when `args` is None.
    RuntimeError
        If acpype executable is not found or if acpype execution fails.
    UserWarning
        If charge cannot be automatically determined from input file.
    
    Notes
    -----
    When `net_charge` is "auto", the function attempts to parse the input
    file using RDKit to extract formal charges. If parsing fails, a warning
    is issued and acpype will determine the charge automatically.
    
    Examples
    --------
    >>> run_acpype("ligand.sdf", basename="LIG", atom_type="gaff2")
    >>> run_acpype("ligand.mol2", charge_method="gas", net_charge=0)
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
    """
    GAFF-based small molecule force field parameterizer.
    
    This class uses ``acpype`` to generate AMBER (prmtop/inpcrd) and GROMACS
    (top) force field parameter files using the General Amber Force Field
    (GAFF or GAFF2) for small molecules.
    
    The parametrization process uses ``acpype`` to assign atom types, generate
    topology, and compute partial charges. The output is validated by
    converting to OpenMM XML format and comparing energies.
    
    Parameters
    ----------
    atype : str, default 'gaff2'
        Atom typing scheme. Must be either 'gaff' or 'gaff2'.
    charge_method : str, default 'bcc'
        Method for partial charge assignment. Must be either 'bcc' (AM1-BCC)
        or 'gas' (Gasteiger).
    reuse_cache : bool, default False
        If True, reuse existing acpype output directory if found. If False,
        remove and regenerate the acpype directory.
    
    Attributes
    ----------
    atype : str
        Atom typing scheme being used.
    charge_method : str
        Charge assignment method being used.
    reuse_cache : bool
        Whether to reuse cached acpype results.
    
    Raises
    ------
    AssertionError
        If `atype` is not 'gaff' or 'gaff2', or if `charge_method` is not
        'bcc' or 'gas'.
    
    Examples
    --------
    >>> gaff = GAFF(atype='gaff2', charge_method='bcc')
    >>> gaff.parametrize('ligand.sdf', wdir='./output')
    
    See Also
    --------
    :func:`run_acpype` : Function that executes acpype.
    :func:`process_prmtop` : Decorator that validates parameter conversion.
    """
    
    def __init__(self, atype: Literal['gaff2', 'gaff'] = 'gaff2', charge_method: Literal['bcc', 'gas'] = 'bcc', reuse_cache: bool = False):
        self.atype = atype
        self.charge_method = charge_method
        assert self.atype in ['gaff', 'gaff2'], f'Unsupported atom type: {atype}'
        assert self.charge_method in ['bcc', 'gas'], f'Unsupported charge method: {atype}'
        self.reuse_cache = reuse_cache
    
    @process_prmtop
    def parametrize(self, ligand_file: os.PathLike, wdir: os.PathLike | None = None):
        """
        Generate GAFF force field parameters for a ligand.
        
        This method runs acpype to generate AMBER and GROMACS topology files
        for the input ligand. The output files are written to the working
        directory with names based on the input ligand file stem.
        
        Parameters
        ----------
        ligand_file : os.PathLike
            Path to input ligand structure file. Must be an SDF file.
        wdir : os.PathLike, optional
            Working directory for output files. If None, uses current directory.
            Output files include:
            
            * ``{stem}.prmtop``: AMBER topology file
            * ``{stem}.inpcrd``: AMBER coordinate file
            * ``{stem}.top``: GROMACS topology file
            * ``{stem}.xml``: OpenMM force field XML (generated by :func:`easybfe.smff.utils.process_prmtop`)
            * ``{stem}.pdb``: PDB structure file (generated by :func:`easybfe.smff.utils.process_prmtop`)
        
        Raises
        ------
        AssertionError
            If `ligand_file` does not have .sdf extension.
        RuntimeError
            If acpype execution fails or if required files are not generated.
        UserWarning
            If existing acpype directory is found and `reuse_cache` is True.
        
        Notes
        -----
        This method is decorated with :func:`easybfe.smff.utils.process_prmtop`, which automatically:
        
        1. Converts the generated prmtop to OpenMM XML format
        2. Validates the conversion by comparing energies
        3. Raises an error if energy difference exceeds 0.01 kJ/mol
        
        The method uses acpype with basename "MOL", so intermediate files
        are created in a "MOL.acpype" subdirectory within the working directory.
        
        See Also
        --------
        :func:`run_acpype` : The underlying acpype execution function.
        """
        ligand_file = Path(ligand_file).resolve()
        assert ligand_file.suffix == '.sdf'
        wdir = Path(wdir).resolve()
        with set_directory(wdir):
            if os.path.isdir('MOL.acpype'):
                if not self.reuse_cache:
                    shutil.rmtree('MOL.acpype')
                else:
                    warnings.warn(f'Found existing MOL.acpype directory, acpype will reuse it.')
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
