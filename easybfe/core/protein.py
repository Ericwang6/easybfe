import os
from pathlib import Path
import io
from pydantic import BaseModel, Field


class Protein(BaseModel):
    name: str
    desc: str = Field(default='protein')
    pdb_string: str

    def stream(self):
        _io = io.StringIO(self.pdb_string)
        _io.seek(0)
        return _io
    
    def to_openmm(self):
        import openmm.app as app
        return app.PDBFile(self.stream())
    
    def dump(self, dirname: os.PathLike):
        Path(dirname).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(dirname, f'{self.name}.pdb'), 'w') as f:
            f.write(self.pdb_string)
    
    @classmethod
    def from_pdb(cls, pdb_path: os.PathLike, name: str | None = None, desc: str = 'protein'):
        pdb_path = Path(pdb_path)
        name = pdb_path.stem if (not name) else name
        return cls(name=name, desc=desc, pdb_string=pdb_path.read_text())
    
