import os
from pathlib import Path
from .base import SmallMoleculeForceField


class CustomForceField(SmallMoleculeForceField):
    def __init__(self, custom_ff: os.PathLike, overwrite: bool = True):
        import parmed
        self.overwrite = overwrite
        self.parmed_struct = parmed.load_file(custom_ff)

    def parametrize(self, ligand_file: os.PathLike, wdir: os.PathLike | None = None):
        prmtop_path = os.path.join(wdir, Path(ligand_file).stem + '.prmtop')
        self.parmed_struct.save(prmtop_path, overwrite=self.overwrite)
