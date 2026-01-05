'''
Author: Eric Wang
Date: 10/07/2024

This file contains abstract class for a small molecule force field parameterizer
'''
import abc
import os
from typing import Optional, Union
from pathlib import Path
import shutil
import logging
from copy import deepcopy

import openmm as mm
import openmm.unit as unit
import openmm.app as app
import parmed
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from .utils import convert_to_xml, read_molecule_from_file


logger = logging.getLogger(__name__)


class SmallMoleculeForceField(abc.ABC):
    """
    Abstract base class for small molecule force field parameterizers.
    
    This class defines the interface that all force field parameterization
    implementations must follow. Subclasses should implement :meth:`_parametrize`
    to generate force field parameters for small molecules.
    
    Examples
    --------
    Subclasses include :class:`easybfe.smff.gaff.GAFF`, 
    :class:`easybfe.smff.openff.OpenFF`, and 
    :class:`easybfe.smff.custom.CustomForceField`.
    """
    
    def __init__(self, forcefield: str = '', charge_method: str = '', *args, **kwargs):
        """
        Initialize the force field parameterizer.
        
        Parameters
        ----------
        forcefield : str, optional
            Force field name or identifier.
        charge_method : str, optional
            Method for assigning partial charges.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self.forcefield = forcefield
        self.charge_method = charge_method

    def _setup(self, ligand: Union[str, Path, Chem.Mol], wdir: os.PathLike, name: Optional[str] = None, overwrite: bool = False):
        """
        Set up the working directory and prepare the ligand molecule.
        
        Parameters
        ----------
        ligand : str, Path, or Chem.Mol
            Path to ligand file, SMILES string, or RDKit molecule object.
        wdir : os.PathLike
            Working directory for output files.
        name : str, optional
            Name for the ligand. Required if ligand is a SMILES string or RDKit molecule.
        overwrite : bool, default False
            Whether to overwrite existing working directory.
        """
        wdir = Path(wdir).resolve()
        if wdir.is_dir() and overwrite:
            logger.info(f"Removing existing directory: {wdir}")
            shutil.rmtree(wdir)
        wdir.mkdir()

        if isinstance(ligand, Chem.Mol):
            # Make a copy to avoid modifying the original molecule
            mol = deepcopy(ligand)
            # Ensure molecule has hydrogens
            if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
                mol = Chem.AddHs(mol)
            # Check if molecule has 3D coordinates
            needs_3d = False
            try:
                conf = mol.GetConformer()
                if conf is None or not conf.Is3D():
                    needs_3d = True
            except (ValueError, RuntimeError):
                # No conformer exists
                needs_3d = True
            if needs_3d:
                logger.info("Generating 3D coordinates for molecule")
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
            assert name, 'Must provide a name when input ligand is an RDKit molecule'
        elif os.path.isfile(ligand):
            mol = read_molecule_from_file(ligand)
            name = Path(ligand).stem if not name else name
            logger.info(f"Loaded molecule from file: {ligand}")
        else:
            logger.info(f"Generating 3D structure from SMILES: {ligand}")
            mol = Chem.AddHs(Chem.MolFromSmiles(ligand))
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            assert name, 'Must provide a name when input ligand is a SMILES string'

        mol.SetProp('_Name', name)
        with Chem.SDWriter(wdir / f'{name}.sdf') as w:
            w.write(mol)
        
        mol_noh = Chem.RemoveHs(mol)
        AllChem.Compute2DCoords(mol_noh)
        Draw.MolToFile(mol_noh, wdir / f'{name}.png', legend=name, size=(500, 500))

        self.file = wdir / f'{name}.sdf'
        self.rdmol = mol
        self.wdir = wdir
        self.name = name

    def _validate(self):
        """
        Validate parametrization by converting to XML and comparing energies.
        
        This method loads the generated prmtop/inpcrd files, converts them to
        OpenMM XML format, and validates the conversion by comparing potential
        energies from both representations.
        
        Raises
        ------
        AssertionError
            If the energy difference between prmtop and XML systems exceeds
            0.01 kJ/mol, indicating an incompatible force field conversion.
        """
        prmtop = str(self.wdir / f'{self.name}.prmtop')
        inpcrd = str(self.wdir / f'{self.name}.inpcrd')
        pdb = str(self.wdir / f'{self.name}.pdb')
        ffxml = str(self.wdir / f'{self.name}.xml')

        logger.info(f"Validating parametrization for {self.name}")
        struct = parmed.load_file(prmtop, xyz=inpcrd)
        struct.residues[0].name = 'MOL'
        app.PDBFile.writeFile(struct.topology, struct.positions, pdb, keepIds=True)
        convert_to_xml(struct, ffxml)

        system_ref = app.AmberPrmtopFile(prmtop).createSystem()
        ctx_ref = mm.Context(system_ref, mm.LangevinIntegrator(300, 1.0, 0.001))
        ctx_ref.setPositions(app.AmberInpcrdFile(inpcrd).positions)
        energy_ref = ctx_ref.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        pdb_obj = app.PDBFile(pdb)
        system = app.ForceField(ffxml).createSystem(pdb_obj.topology)
        ctx = mm.Context(system, mm.LangevinIntegrator(300, 1.0, 0.001))
        ctx.setPositions(app.AmberInpcrdFile(inpcrd).positions)
        energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        energy_diff = abs(energy_ref - energy)
        if energy_diff >= 0.01:
            logger.error(f"Energy validation failed: {energy_ref} != {energy} (diff: {energy_diff:.6f} kJ/mol)")
            raise AssertionError(
                f"Fail to convert prmtop to xml, the force field might not be compatitable "
                f"because the energy is different {energy_ref} != {energy}"
            )
        logger.info(f"Energy validation passed.")
        logger.debug(f"Energy {energy_ref:.4f} kJ/mol with prmtop and {energy:.4f} with xml")

    def run(self, ligand: Union[str, Path, Chem.Mol], wdir: os.PathLike, name: Optional[str] = None, overwrite: bool = False):
        """
        Run the complete parametrization workflow.
        
        This method orchestrates the parametrization process by calling
        :meth:`_setup`, :meth:`_parametrize`, and :meth:`_validate` in sequence.
        
        Parameters
        ----------
        ligand : str, Path, or Chem.Mol
            Path to ligand file, SMILES string, or RDKit molecule object.
        wdir : os.PathLike
            Working directory for output files.
        name : str, optional
            Name for the ligand. Required if ligand is a SMILES string or RDKit molecule.
        overwrite : bool, default False
            Whether to overwrite existing working directory.
        """
        self._setup(ligand, wdir, name, overwrite)
        self._parametrize()
        self._validate()

    @abc.abstractmethod
    def _parametrize(self):
        """
        Generate force field parameters for a ligand molecule.
        
        This method should generate force field parameter files (e.g., prmtop,
        inpcrd) for the ligand. The method has access to the following attributes
        set by :meth:`_setup`:
        
        * ``self.file``: Path to the ligand SDF file
        * ``self.rdmol``: RDKit molecule object
        * ``self.wdir``: Working directory
        * ``self.name``: Ligand name
        
        Notes
        -----
        The output files should be written to ``self.wdir`` with names based on
        ``self.name``. At minimum, the following files should be generated:
        
        * ``{self.name}.prmtop``: AMBER topology file
        * ``{self.name}.inpcrd``: AMBER coordinate file
        """
        ...