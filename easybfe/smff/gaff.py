'''
Author: Eric Wang
Date: 10/07/2024

This file contains GAFF-based small molecule force field parameterizer
'''
import warnings
import os, shutil
from pathlib import Path
from typing import List, Union, Optional, Literal
import logging
import parmed
from rdkit import Chem

from .utils import read_molecule_from_file
from .base import SmallMoleculeForceField
from ..cmd import find_executable, run_command, set_directory


logger = logging.getLogger(__name__)


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
        logger.info(f"Running acpype with custom arguments")
    else:
        assert input is not None, "Input is None."
        cmd = [acpype, "-i", str(input), "-b", basename, "-c", charge_method, "-a", atom_type]            
        if net_charge == "auto":
            try:
                mol = read_molecule_from_file(input)
                net_charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])
                logger.info(f"Auto-detected net charge: {net_charge}")
            except:
                warnings.warn(f"Fail to parse input file {Path(input).resolve()}. Will use acpype to determine net charge")
        if not isinstance(net_charge, str):
            cmd.extend(["-n", str(net_charge)])
        logger.info(f"Running acpype: {' '.join(cmd)}")
    return_code, out, err = run_command(cmd, raise_error=True)
    return 


class GAFF(SmallMoleculeForceField):
    """
    GAFF-based small molecule force field parameterizer.
    
    This class uses ``acpype`` to generate AMBER (prmtop/inpcrd) and GROMACS
    (top) force field parameter files using the General Amber Force Field
    (GAFF or GAFF2) for small molecules.
    
    Parameters
    ----------
    forcefield : {'gaff2', 'gaff'}, default 'gaff2'
        Atom typing scheme.
    charge_method : {'bcc', 'gas'}, default 'bcc'
        Method for partial charge assignment. 'bcc' uses AM1-BCC charges,
        'gas' uses Gasteiger charges.
    reuse_cache : bool, default False
        If True, reuse existing acpype output directory if found. If False,
        remove and regenerate the acpype directory.
    
    Attributes
    ----------
    reuse_cache : bool
        Whether to reuse cached acpype results.
    
    Raises
    ------
    AssertionError
        If `forcefield` is not 'gaff' or 'gaff2', or if `charge_method` is not
        'bcc' or 'gas'.
    
    Examples
    --------
    >>> gaff = GAFF(forcefield='gaff2', charge_method='bcc')
    >>> gaff.run('ligand.sdf', wdir='./output')
    
    See Also
    --------
    :func:`run_acpype` : Function that executes acpype.
    """
    
    def __init__(self, forcefield: Literal['gaff2', 'gaff'] = 'gaff2', charge_method: Literal['bcc', 'gas'] = 'bcc', reuse_cache: bool = False):
        super().__init__(forcefield, charge_method)

        assert self.forcefield in ['gaff', 'gaff2'], f'Unsupported atom type: {forcefield}'
        assert self.charge_method in ['bcc', 'gas'], f'Unsupported charge method: {charge_method}'
        self.reuse_cache = reuse_cache
        logger.info(f"Initialized GAFF with forcefield={forcefield}, charge_method={charge_method}")
    
    def _parametrize(self):
        """
        Generate GAFF force field parameters for a ligand.
        
        This method runs acpype to generate AMBER and GROMACS topology files
        for the input ligand. The output files are written to the working
        directory with names based on the ligand name.
        
        Raises
        ------
        RuntimeError
            If acpype execution fails or if required files are not generated.
        UserWarning
            If existing acpype directory is found and `reuse_cache` is True.
        
        Notes
        -----
        The method uses acpype with basename ``{self.name}``, so intermediate files
        are created in a ``{self.name}.acpype`` subdirectory within the working directory.
        """
        ligand_file = self.file
        wdir = self.wdir
        stem = self.name

        logger.info(f"Generating GAFF parameters for {stem}")
        with set_directory(wdir):
            if os.path.isdir(f'{stem}.acpype'):
                if not self.reuse_cache:
                    logger.info(f"Removing existing acpype directory: {stem}.acpype")
                    shutil.rmtree(f'{stem}.acpype')
                else:
                    warnings.warn(f'Found existing {stem}.acpype directory, acpype will reuse it.')
            run_acpype(
                ligand_file,
                basename=stem,
                charge_method=self.charge_method,
                atom_type=self.forcefield,
                net_charge='auto'
            )
            # amber format
            prmtop_src = f'{stem}.acpype/{stem}_AC.prmtop'
            prmtop_dst = f'{stem}.prmtop'
            logger.info(f"Copying {prmtop_src} to {prmtop_dst}")
            shutil.copyfile(prmtop_src, prmtop_dst)
            struct = parmed.load_file(prmtop_src)
            struct.coordinates = Chem.SDMolSupplier(str(ligand_file), removeHs=False)[0].GetConformer().GetPositions()
            struct.save(f'{stem}.inpcrd', overwrite=True)
            # gmx format
            top_file = f'{stem}.top'
            logger.info(f"Generating GROMACS topology: {top_file}")
            with open(top_file, 'w') as fp:
                with open(f'{stem}.acpype/{stem}_GMX.itp') as f:
                    itp = f.read()
                with open(f'{stem}.acpype/{stem}_GMX.top') as f:
                    for line in f:
                        if line.startswith(f'#include "{stem}_GMX.itp"'):
                            fp.write(itp)
                            fp.write('\n')
                        elif line.startswith('#'):
                            continue
                        else:
                            fp.write(line)
        logger.info(f"Completed GAFF parametrization for {stem}")
