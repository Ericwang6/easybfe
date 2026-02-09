"""
GAFF and GAFF2 force field parameterization using acpype.

This module provides an interface to the acpype tool for generating AMBER and
GROMACS topology files using the General Amber Force Field (GAFF or GAFF2).
"""
from __future__ import annotations
from typing import TYPE_CHECKING
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
if TYPE_CHECKING:
    from ..core.ligand import Ligand


logger = logging.getLogger(__name__)


def run_acpype(input: Optional[os.PathLike] = None,
               basename: str = "MOL",
               charge_method: str = "bcc",
               atom_type: str = "gaff2",
               net_charge: Union[int, str] = "auto",
               args: Union[None, List[str]] = None):
    """
    Execute acpype to generate AMBER and GROMACS topology files.
    
    Provides a Python interface to the acpype command-line tool, which uses
    Antechamber to generate force field parameters for small molecules using
    GAFF or GAFF2 atom types.
    
    Parameters
    ----------
    input : os.PathLike, optional
        Input structure file (.mol, .sdf, .mol2, .pdb). Required unless `args`
        is provided.
    basename : str, default='MOL'
        Output file basename. Acpype creates ``{basename}.acpype/`` directory
        with files like ``{basename}_AC.prmtop``.
    charge_method : str, default='bcc'
        Partial charge method:
        
        * 'bcc': AM1-BCC charges (semi-empirical, accurate)
        * 'gas': Gasteiger charges (fast, approximate)
        * 'user': Read charges from input mol2 file
    atom_type : str, default='gaff2'
        Atom typing scheme:
        
        * 'gaff': General Amber Force Field v1
        * 'gaff2': GAFF2 (improved parameters, recommended)
        * 'amber': AMBER protein force field (for peptides)
        * 'amber2': AMBER + GAFF2 (for protein-ligand complexes)
    net_charge : int or str, default='auto'
        Molecular net charge:
        
        * 'auto': Auto-detect from RDKit formal charges
        * int: Explicit charge value (e.g., 0, +1, -1)
        * Not specified: Let acpype determine charge
    args : list of str, optional
        Raw acpype command-line arguments. If provided, all other parameters
        are ignored.
    
    Raises
    ------
    AssertionError
        If `input` is None when `args` is None.
    RuntimeError
        If acpype executable not found or execution fails.
    UserWarning
        If charge auto-detection fails for 'auto' mode.
    
    Notes
    -----
    Acpype generates files in ``{basename}.acpype/`` directory:
    
    * ``{basename}_AC.prmtop``, ``{basename}_AC.inpcrd``: AMBER format
    * ``{basename}_GMX.top``, ``{basename}_GMX.itp``: GROMACS format
    * ``{basename}_bcc.mol2``: Mol2 with assigned charges
    
    When `net_charge='auto'`, RDKit is used to sum formal charges from the
    input molecule. If RDKit cannot parse the file, acpype will attempt to
    determine the charge automatically.
    
    Examples
    --------
    >>> # Basic usage with auto-detected charge
    >>> run_acpype('ligand.sdf', basename='LIG')
    >>> # Specify charge explicitly
    >>> run_acpype('ligand.mol2', basename='LIG', net_charge=-1)
    >>> # Use Gasteiger charges (faster)
    >>> run_acpype('ligand.sdf', charge_method='gas', atom_type='gaff2')
    
    See Also
    --------
    :class:`GAFF` : High-level GAFF parameterizer class.
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
    GAFF/GAFF2 parameterizer using acpype wrapper.
    
    Generates AMBER (prmtop/inpcrd) and GROMACS (top) topology files for small
    molecules using the General Amber Force Field (GAFF or GAFF2) via the
    acpype tool. Supports AM1-BCC or Gasteiger charge assignment.
    
    Parameters
    ----------
    forcefield : {'gaff2', 'gaff'}, default='gaff2'
        Atom typing scheme. GAFF2 is recommended for improved parameters.
    charge_method : {'bcc', 'gas'}, default='bcc'
        Partial charge method:
        
        * 'bcc': AM1-BCC (accurate, semi-empirical, slower)
        * 'gas': Gasteiger (fast, empirical, less accurate)
    reuse_cache : bool, default=False
        Whether to reuse existing ``{name}.acpype/`` directory if present.
        If False, the directory is removed and acpype re-runs.
    
    Attributes
    ----------
    reuse_cache : bool
        Cache reuse setting.
    
    Raises
    ------
    AssertionError
        If `forcefield` not in ['gaff', 'gaff2'] or `charge_method` not in
        ['bcc', 'gas'].
    RuntimeError
        If acpype execution fails.
    
    Notes
    -----
    This class requires the acpype command-line tool to be installed and
    available in PATH. Acpype internally uses Antechamber and other AmberTools
    programs.
    
    Generated files:
    
    * ``{name}.prmtop``: AMBER topology (copied from acpype output)
    * ``{name}.inpcrd``: AMBER coordinates (coordinates from input SDF)
    * ``{name}.top``: GROMACS topology (merged from .itp and .top)
    * ``{name}.acpype/``: Acpype working directory (unless `reuse_cache=True`)
    
    Examples
    --------
    >>> from easybfe.ligand import LigandLoader
    >>> # GAFF2 with AM1-BCC charges
    >>> gaff = GAFF(forcefield='gaff2', charge_method='bcc')
    >>> loader = LigandLoader()
    >>> ligands = loader.load('ligand.sdf', only_first=True)
    >>> ligand = gaff.run(ligands[0])
    >>> # GAFF with Gasteiger charges (faster)
    >>> gaff_fast = GAFF(forcefield='gaff', charge_method='gas')
    >>> ligand = gaff_fast.run(ligands[0])
    
    See Also
    --------
    :func:`run_acpype` : Low-level acpype execution function.
    :class:`easybfe.smff.openff.OpenFF` : OpenFF alternative.
    """
    
    def __init__(self, forcefield: Literal['gaff2', 'gaff'] = 'gaff2', charge_method: Literal['bcc', 'gas'] = 'bcc'):
        super().__init__(forcefield, charge_method)

        assert self.forcefield in ['gaff', 'gaff2'], f'Unsupported atom type: {forcefield}'
        assert self.charge_method in ['bcc', 'gas'], f'Unsupported charge method: {charge_method}'
        logger.info(f"Initialized GAFF with forcefield={forcefield}, charge_method={charge_method}")

        self.reuse_cache = False
    
    def _parametrize(self, ligand: Ligand, wdir: str):
        """
        Generate GAFF parameters via acpype.
        
        Executes acpype to assign atom types, generate bonded parameters, and
        compute partial charges using the specified method. Produces AMBER and
        GROMACS topology files.
        
        Parameters
        ----------
        ligand : Ligand
            Ligand object with 3D structure in mol_block.
        wdir : str
            Working directory path for writing output files.
        
        Returns
        -------
        Ligand
            Ligand object with prmtop, inpcrd, and top files stored as auxiliary files.
        
        Raises
        ------
        RuntimeError
            If acpype execution fails.
        UserWarning
            If ``{name}.acpype/`` directory exists and `reuse_cache=True`.
        
        Notes
        -----
        Workflow:
        
        1. Write ligand mol_block to ``{ligand.name}.mol`` in wdir
        2. Check for existing ``{ligand.name}.acpype/`` directory (remove or reuse)
        3. Run acpype with auto-detected net charge from RDKit
        4. Copy ``{ligand.name}_AC.prmtop`` to ``{ligand.name}.prmtop``
        5. Generate ``{ligand.name}.inpcrd`` with coordinates from mol_block
        6. Merge GROMACS ``.itp`` and ``.top`` files into ``{ligand.name}.top``
        7. Store prmtop, inpcrd, and top files as auxiliary files in ligand object
        
        The GROMACS topology file has the .itp file contents inlined and
        comment lines removed for easier distribution as a single file.
        
        See Also
        --------
        :func:`run_acpype` : Underlying acpype execution function.
        """
        ligand_file = os.path.join(wdir, f'{ligand.name}.mol')
        with open(ligand_file, 'w') as f:
            f.write(ligand.mol_block)
        stem = ligand.name

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

            with open(prmtop_dst) as f:
                ligand.add_aux_file('prmtop', f.read())
            with open(f'{stem}.inpcrd') as f:
                ligand.add_aux_file('inpcrd', f.read())
            with open(top_file) as f:
                ligand.add_aux_file('top', f.read())
        logger.info(f"Completed GAFF parametrization for {stem}")
        return ligand
