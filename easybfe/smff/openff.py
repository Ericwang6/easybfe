import os
from pathlib import Path
from rdkit import Chem
import logging
import parmed
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import openmm.app as app

from .base import SmallMoleculeForceField


logger = logging.getLogger(__name__)


class OpenFF(SmallMoleculeForceField):
    """
    OpenFF-based small molecule force field parameterizer.
    
    This class generates force field parameters using the Open Force Field
    (OpenFF) initiative's SMIRNOFF-based force fields. It uses the OpenFF
    toolkit to assign parameters and generate AMBER and GROMACS topology files.
    
    Parameters
    ----------
    forcefield : str, default 'openff-2.1.0'
        Name of the OpenFF force field to use (e.g., 'openff-2.1.0',
        'openff-2.0.0', 'openff-1.3.0').
    charge_method : str, default 'bcc'
        Method for partial charge assignment. Options:
        
        * 'bcc': AM1-BCC charges (mapped to 'am1bcc' in OpenFF)
        * 'gas': Gasteiger charges (mapped to 'gasteiger' in OpenFF)
        * Other: Passed directly to OpenFF toolkit
    
    Attributes
    ----------
    top : openmm.app.Topology
        OpenMM topology object (set after parametrization).
    system : openmm.System
        OpenMM system object (set after parametrization).
    struct : parmed.Structure
        ParmEd structure object (set after parametrization).
    
    Examples
    --------
    >>> openff = OpenFF(forcefield='openff-2.1.0', charge_method='bcc')
    >>> openff.run('ligand.sdf', wdir='./output')
    >>> # Access generated structure
    >>> struct = openff.struct
    
    See Also
    --------
    :class:`easybfe.smff.gaff.GAFF` : GAFF-based parameterizer.
    :class:`easybfe.smff.custom.CustomForceField` : Custom force field parameterizer.
    """
    
    def __init__(self, forcefield: str = 'openff-2.1.0', charge_method: str = 'bcc'):
        super().__init__(forcefield, charge_method)
        
        if charge_method == 'gas':
            self.charge_method = 'gasteiger'
        elif charge_method == 'bcc':
            self.charge_method = 'am1bcc'
        else:
            self.charge_method = charge_method
        logger.info(f"Initialized OpenFF with forcefield={forcefield}, charge_method={self.charge_method}")
        
    def _parametrize(self):
        """
        Generate OpenFF force field parameters for a ligand.
        
        This method uses the OpenFF toolkit to assign SMIRNOFF-based force
        field parameters to the ligand molecule. It generates AMBER and
        GROMACS topology files, as well as OpenMM system objects.
        
        Raises
        ------
        RuntimeError
            If molecule parsing fails, stereochemistry assignment fails, or
            if parameter assignment fails.
        
        Notes
        -----
        This method:
        
        1. Assigns stereochemistry using RDKit
        2. Converts to OpenFF Molecule object
        3. Assigns partial charges using the specified method
        4. Generates SMIRNOFF parameters using the specified force field
        5. Creates OpenMM system and ParmEd structure
        6. Saves topology files in multiple formats
        
        The method sets `self.top`, `self.system`, and `self.struct` attributes
        after successful parametrization.
        """
        mol = self.rdmol
        wdir = self.wdir
        stem = self.name

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
        struct.save(str(wdir / f'{stem}.top'), overwrite=True, combine='all')
        struct.save(str(wdir / f'{stem}.prmtop'), overwrite=True)
        struct.save(str(wdir / f'{stem}.inpcrd'), overwrite=True)
        self.top = top
        self.system = system
        self.struct = struct
        logger.info(f"Completed OpenFF parametrization for {stem}")
        