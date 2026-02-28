from __future__ import annotations
from typing import TYPE_CHECKING
import os
from rdkit import Chem
import parmed
import logging

from .base import SmallMoleculeForceField
from .registry import PARAMETRIZER_REGISTRY

if TYPE_CHECKING:
    from ..core.ligand import Ligand


logger = logging.getLogger(__name__)


@PARAMETRIZER_REGISTRY.register("custom")
class CustomForceField(SmallMoleculeForceField):
    """
    Parameterizer using pre-existing force field topology.
    
    This class reuses topology and parameters from a pre-parameterized structure
    file, applying them to new molecular coordinates. This is useful for:
    
    * Using the same parameters across multiple conformations
    * Applying pre-validated parameters from external sources
    * Working with force fields not directly supported
    
    The topology (atom types, bonds, angles, dihedrals, nonbonded parameters)
    is preserved from the custom force field file, while coordinates are taken
    from the input ligand.
    
    Parameters
    ----------
    custom_ff : os.PathLike
        Path to pre-parameterized structure file that ParmEd can load
        (e.g., .prmtop, .top, .gro, .psf). Topology and parameters will be
        extracted from this file.
    charge_method : str, optional
        Charge method (kept for API compatibility, not used).
    overwrite : bool, default=True
        Whether to overwrite existing output files.
    
    Attributes
    ----------
    overwrite : bool
        Whether to overwrite existing files.
    parmed_struct : parmed.Structure
        ParmEd structure loaded from `custom_ff`.
    
    Raises
    ------
    RuntimeError
        If the number of atoms in the input ligand does not match the topology.
    
    Examples
    --------
    >>> from easybfe.ligand import LigandLoader
    >>> # Use topology from a prmtop file with new coordinates
    >>> custom = CustomForceField('reference.prmtop')
    >>> loader = LigandLoader()
    >>> ligands = loader.load('new_conformation.sdf', only_first=True)
    >>> ligand = custom.run(ligands[0])
    
    See Also
    --------
    :class:`easybfe.smff.gaff.GAFF` : Generate parameters using GAFF/GAFF2.
    :class:`easybfe.smff.openff.OpenFF` : Generate parameters using OpenFF.
    """
    
    def __init__(self, custom_ff: os.PathLike, charge_method: str = '', *args, **kwargs):
        super().__init__(custom_ff, charge_method, *args, **kwargs)
        logger.info(f"Loading custom force field from: {custom_ff}")
        self.parmed_struct = parmed.load_file(custom_ff)
    
    def _parametrize(self, ligand: Ligand, wdir: str):
        """
        Apply pre-existing topology to new ligand coordinates.
        
        Extracts coordinates from the input ligand mol_block and combines them
        with the topology/parameters from the custom force field. Generates
        prmtop and inpcrd files with the same force field parameters but new
        atomic positions.
        
        Parameters
        ----------
        ligand : Ligand
            Ligand object with 3D structure in mol_block.
        wdir : str
            Working directory path for writing output files.
        
        Returns
        -------
        Ligand
            Ligand object with prmtop and inpcrd files stored as auxiliary files.
        
        Raises
        ------
        RuntimeError
            If the input ligand cannot be read from the mol_block.
        AssertionError
            If the number of atoms in the input ligand does not match the
            number of atoms in the custom force field topology.
        
        Notes
        -----
        The input ligand must have the same atom ordering as the reference
        structure used to create the custom force field file.
        
        Workflow:
        
        1. Extract coordinates from ``ligand.get_rdmol()``
        2. Apply coordinates to pre-loaded ParmEd structure
        3. Save prmtop and inpcrd files to wdir
        4. Store files as auxiliary files in ligand object
        """
        prmtop_path = os.path.join(wdir, f'{ligand.name}.prmtop')
        inpcrd_path = os.path.join(wdir, f'{ligand.name}.inpcrd')
        
        positions = ligand.get_rdmol().GetConformer().GetPositions()
        self.parmed_struct.coordinates = positions
        self.parmed_struct.save(str(prmtop_path))
        self.parmed_struct.save(str(inpcrd_path))
        logger.info(f"Saved prmtop and inpcrd files for {ligand.name}")

        with open(prmtop_path) as f:
            ligand.add_aux_file('prmtop', f.read())
        with open(inpcrd_path) as f:
            ligand.add_aux_file('inpcrd', f.read())
        return ligand