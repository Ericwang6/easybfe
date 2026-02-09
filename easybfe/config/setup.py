from typing import Literal, Optional, Iterable
from pydantic import BaseModel, Field, field_validator, model_validator


FF_QUICK_MAP = {
    "ff14sb": "amber14-all.xml",
    "tip3p": "amber14/tip3p.xml"
}


def standarize_ff(ffstr: str) -> str:
    return FF_QUICK_MAP.get(ffstr.lower(), ffstr)


class SetupConfig(BaseModel):
    # setup settings
    box_shape: Literal['cube', 'dodecahedron', 'octahedron'] = 'cube'
    buffer: float = 20.0
    neutralize: bool = True
    ionic_strength: float = 0.15
    do_hmr: bool = True
    hydrogen_mass: float = 3.024
    gas_phase: bool = False
    water_model: Optional[Literal['tip3p', 'spce', 'tip4pew', 'tip5p', 'swm4ndp']] = 'tip3p'

    protein_ff: str | list[str] = Field(default_factory=lambda: ['ff14sb'])
    water_ff: str | list[str] = Field(default_factory=lambda: ['tip3p'])
    extra_ff: str | list[str] = Field(default_factory=list)

    forcefields: list[str] = Field(init=False, exclude=True, default_factory=list)
    
    @model_validator(mode='after')
    def validate_force_field(self):

        def _stdffs(ffs):
            if isinstance(ffs, str):
                ffs = [standarize_ff(ffs)]
            elif ffs is None:
                ffs = []
            else:
                ffs = [standarize_ff(x) for x in ffs]
            return ffs
        
        self.protein_ff = _stdffs(self.protein_ff)
        self.water_ff = _stdffs(self.water_ff)
        self.extra_ff = _stdffs(self.extra_ff)
        self.forcefields = self.protein_ff + self.water_ff + self.extra_ff
        return self
    
    #TODO: automatic infer water model from water_ff
