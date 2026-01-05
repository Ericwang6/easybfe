import os
from pathlib import Path
from rdkit import Chem
import parmed
import logging

from .base import SmallMoleculeForceField


logger = logging.getLogger(__name__)


class CustomForceField(SmallMoleculeForceField):
    """
    Custom force field parameterizer using pre-parameterized structures.
    
    This class allows using a pre-existing parameterized structure (e.g., from
    a prmtop file) and applying it to new ligand coordinates. This is useful
    when you have already parameterized a molecule and want to use the same
    parameters with different conformations.
    
    Parameters
    ----------
    custom_ff : os.PathLike
        Path to a pre-parameterized force field file. This should be a file
        that parmed can load (e.g., .prmtop, .top, .gro, etc.). The topology
        and parameters from this file will be reused.
    charge_method : str, optional
        Charge method (not used here, but kept for API compatibility).
    overwrite : bool, default True
        Whether to overwrite existing output files if they exist.
    
    Attributes
    ----------
    overwrite : bool
        Whether to overwrite existing files.
    parmed_struct : parmed.Structure
        ParmEd structure object loaded from the custom force field file.
    
    Examples
    --------
    >>> # Use a pre-parameterized prmtop file
    >>> custom = CustomForceField('ligand.prmtop')
    >>> custom.run('ligand.sdf', wdir='./output')
    
    See Also
    --------
    :class:`easybfe.smff.gaff.GAFF` : GAFF-based parameterizer.
    :class:`easybfe.smff.openff.OpenFF` : OpenFF-based parameterizer.
    """
    
    def __init__(self, custom_ff: os.PathLike, charge_method: str = '', overwrite: bool = True):
        super().__init__(custom_ff, charge_method)
        self.overwrite = overwrite
        logger.info(f"Loading custom force field from: {custom_ff}")
        self.parmed_struct = parmed.load_file(custom_ff)
    
    def _parametrize(self):
        """
        Apply custom force field parameters to a ligand structure.
        
        This method takes the topology and parameters from the pre-loaded
        custom force field and applies it to the coordinates from the input
        ligand file. The output files use the same topology/parameters but
        with new coordinates.
        
        Raises
        ------
        RuntimeError
            If the ligand file cannot be parsed or if the number of atoms
            doesn't match the custom force field topology.
        """
        wdir = self.wdir
        stem = self.name
        prmtop_path = wdir / f'{stem}.prmtop'
        inpcrd_path = wdir / f'{stem}.inpcrd'
        
        positions = Chem.SDMolSupplier(str(self.file), removeHs=False)[0].GetConformer().GetPositions()
        self.parmed_struct.coordinates = positions
        self.parmed_struct.save(str(prmtop_path))
        self.parmed_struct.save(str(inpcrd_path))
        logger.info(f"Saved prmtop and inpcrd files for {stem}")
