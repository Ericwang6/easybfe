import os
from pathlib import Path
from rdkit import Chem
from .base import SmallMoleculeForceField


class CustomForceField(SmallMoleculeForceField):
    def __init__(self, custom_ff: os.PathLike, overwrite: bool = True):
        import parmed
        self.overwrite = overwrite
        self.parmed_struct = parmed.load_file(custom_ff)

    def parametrize(self, ligand_file: os.PathLike, wdir: os.PathLike | None = None):
        prmtop_path = os.path.join(wdir, Path(ligand_file).stem + '.prmtop')
        inpcrd_path = os.path.join(wdir, Path(ligand_file).stem + '.inpcrd')
        positions = Chem.SDMolSupplier(ligand_file, removeHs=False)[0].GetConformer().GetPositions()
        self.parmed_struct.coordinates = positions
        self.parmed_struct.save(prmtop_path, overwrite=self.overwrite)
        self.parmed_struct.save(inpcrd_path, overwrite=self.overwrite)
