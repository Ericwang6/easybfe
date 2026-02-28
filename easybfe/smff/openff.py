from __future__ import annotations
from typing import TYPE_CHECKING
import os
from pathlib import Path
from rdkit import Chem
import logging
import parmed
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import openmm.app as app

from .base import SmallMoleculeForceField
from .registry import PARAMETRIZER_REGISTRY

if TYPE_CHECKING:
    from ..core.ligand import Ligand


logger = logging.getLogger(__name__)


@PARAMETRIZER_REGISTRY.register("openff")
class OpenFF(SmallMoleculeForceField):
    """
    OpenFF SMIRNOFF-based parameterizer.
    
    Generates force field parameters using the Open Force Field (OpenFF)
    Initiative's SMIRNOFF format force fields. Leverages the OpenFF toolkit
    for SMARTS-based parameter assignment, producing AMBER (prmtop/inpcrd) and
    GROMACS (top) topology files.
    
    Parameters
    ----------
    forcefield : str, default='openff-2.1.0'
        OpenFF force field identifier. Examples:
        
        * 'openff-2.1.0': Latest recommended OpenFF version
        * 'openff-2.0.0': Previous stable release
        * 'openff_unconstrained-2.1.0': Version without constraints
        * 'openff-1.3.1': Legacy version (Parsley)
    charge_method : str, default='bcc'
        Partial charge method:
        
        * 'bcc': AM1-BCC (mapped to 'am1bcc' in OpenFF)
        * 'gas': Gasteiger (mapped to 'gasteiger' in OpenFF)
        * 'am1bcc', 'am1bccelf10', 'gasteiger': Direct OpenFF names
    
    Attributes
    ----------
    top : openmm.app.Topology
        OpenMM topology (available after :meth:`_parametrize`).
    system : openmm.System
        OpenMM system with force objects (available after :meth:`_parametrize`).
    struct : parmed.Structure
        ParmEd structure (available after :meth:`_parametrize`).
    
    Raises
    ------
    RuntimeError
        If OpenFF toolkit fails to assign parameters or charges.
    
    Notes
    -----
    Requires the following packages:
    
    * ``openff-toolkit``: OpenFF parameterization engine
    * ``openmmforcefields``: SMIRNOFF template generator
    * ``ambertools`` or ``openeye-toolkits``: For AM1-BCC charges
    
    Generated files:
    
    * ``{name}.prmtop``: AMBER topology
    * ``{name}.inpcrd``: AMBER coordinates
    * ``{name}.top``: GROMACS topology (all-in-one format)
    
    Examples
    --------
    >>> from easybfe.ligand import LigandLoader
    >>> # Use latest OpenFF with AM1-BCC charges
    >>> openff = OpenFF('openff-2.1.0', 'bcc')
    >>> loader = LigandLoader()
    >>> ligands = loader.load('ligand.sdf', only_first=True)
    >>> ligand = openff.run(ligands[0])
    
    See Also
    --------
    :class:`easybfe.smff.gaff.GAFF` : GAFF/GAFF2 alternative.
    :class:`easybfe.smff.custom.CustomForceField` : Custom topology reuse.
    """
    
    def __init__(self, forcefield: str = 'openff-2.1.0', charge_method: str = 'bcc', *args, **kwargs):
        super().__init__(forcefield, charge_method, *args, **kwargs)
        
        if charge_method == 'gas':
            self.charge_method = 'gasteiger'
        elif charge_method == 'bcc':
            self.charge_method = 'am1bcc'
        else:
            self.charge_method = charge_method
        logger.info(f"Initialized OpenFF with forcefield={forcefield}, charge_method={self.charge_method}")
        
    def _parametrize(self, ligand: Ligand, wdir: str):
        """
        Generate OpenFF SMIRNOFF parameters.
        
        Uses OpenFF toolkit to assign SMARTS-based parameters and compute
        partial charges. Produces OpenMM system, ParmEd structure, and writes
        AMBER/GROMACS topology files.
        
        Parameters
        ----------
        ligand : Ligand
            Ligand object with input molecule.
        wdir : str
            Working directory path for output files.
        
        Returns
        -------
        Ligand
            Ligand object with parametrized topology files added as auxiliary files.
        
        Raises
        ------
        RuntimeError
            If stereochemistry assignment, molecule conversion, charge
            computation, or parameter assignment fails.
        
        Notes
        -----
        Workflow:
        
        1. Assign stereochemistry to RDKit molecule (force=True)
        2. Convert RDKit Mol to OpenFF Molecule
        3. Compute partial charges using specified method
        4. Create SMIRNOFF template generator
        5. Register generator with OpenMM ForceField
        6. Generate OpenMM System
        7. Convert to ParmEd Structure
        8. Standardize residue name to 'MOL'
        9. Write GROMACS .top, AMBER .prmtop/.inpcrd
        10. Store files as auxiliary files in ligand object
        
        See Also
        --------
        :class:`openff.toolkit.Molecule` : OpenFF molecule representation.
        :class:`openmmforcefields.generators.SMIRNOFFTemplateGenerator` : SMIRNOFF generator.
        """
        mol = ligand.get_rdmol()
        stem = ligand.name

        logger.info(f"Generating OpenFF parameters for {stem}")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        logger.info(f"Assigning partial charges using {self.charge_method}")
        off_mol.assign_partial_charges(self.charge_method, use_conformers=off_mol.conformers)
        generator = SMIRNOFFTemplateGenerator(molecules=[off_mol], forcefield=self.forcefield).generator
        ff = app.ForceField()
        ff.registerTemplateGenerator(generator)
        top = Topology.from_molecules(off_mol).to_openmm()
        system = ff.createSystem(top, constraints=None, rigidWater=False)
        struct = parmed.openmm.load_topology(top, system, xyz=mol.GetConformer().GetPositions())
        struct.residues[0].name = 'MOL'
        
        top_path = os.path.join(wdir, f'{stem}.top')
        prmtop_path = os.path.join(wdir, f'{stem}.prmtop')
        inpcrd_path = os.path.join(wdir, f'{stem}.inpcrd')
        
        struct.save(top_path, overwrite=True, combine='all')
        struct.save(prmtop_path, overwrite=True)
        struct.save(inpcrd_path, overwrite=True)
        
        with open(prmtop_path) as f:
            ligand.add_aux_file('prmtop', f.read())
        with open(inpcrd_path) as f:
            ligand.add_aux_file('inpcrd', f.read())
        with open(top_path) as f:
            ligand.add_aux_file('top', f.read())
        
        logger.info(f"Completed OpenFF parametrization for {stem}")
        return ligand
        