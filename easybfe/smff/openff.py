import os
from pathlib import Path
from rdkit import Chem
import parmed
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
import openmm.app as app

from .base import SmallMoleculeForceField


class OpenFF(SmallMoleculeForceField):
    def __init__(self, forcefield: str = 'openff-2.1.0', charge_method: str = 'bcc'):
        self.forcefield = forcefield

        if charge_method == 'gas':
            self.charge_method = 'gasteiger'
        elif charge_method == 'bcc':
            self.charge_method = 'am1bcc'
        else:
            self.charge_method = charge_method

    def parametrize(self, ligand_file: os.PathLike, wdir: os.PathLike | None = None):
        ligand_file = Path(ligand_file).resolve()
        wdir = Path(wdir).resolve()
        suffix = ligand_file.suffix
        if suffix == ".mol":
            mol = Chem.MolFromMolFile(str(ligand_file), removeHs=False)
        elif suffix == ".sdf":
            mol = Chem.SDMolSupplier(str(ligand_file), removeHs=False)[0]
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False)
        else:
            mol = None
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        off_mol.assign_partial_charges(self.charge_method)
        generator = SMIRNOFFTemplateGenerator(molecules=[off_mol], forcefield=self.forcefield).generator
        ff = app.ForceField()
        ff.registerTemplateGenerator(generator)
        top = Topology.from_molecules(off_mol).to_openmm()
        system = ff.createSystem(top, constraints=None, rigidWater=False)
        struct = parmed.openmm.load_topology(top, system)
        struct.residues[0].name = 'MOL'
        struct.save(str(wdir / f'{ligand_file.stem}.top'), overwrite=True)
        struct.save(str(wdir / f'{ligand_file.stem}.prmtop'), overwrite=True)
        