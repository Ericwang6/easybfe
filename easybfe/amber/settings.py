
from pydantic import BaseModel, Field, model_validator, field_serializer, field_validator, AfterValidator
from typing import Dict, Any
try:
    from typing_extensions import Self, Annotated
except ImportError:
    from typing import Self, Annotated


class AmberNamelist(BaseModel):
    name: str
    sep: str = "\n"
    extra: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @staticmethod
    def write_namelist(name: str, content: Dict[str, Any], sep: str = '\n'):
        assert sep in ['\n', ' ']
        if sep == '\n':
            num_space = max([len(key) for key in content.keys()]) + 1
            template = '{:<' + str(num_space) + '} = {},'
        else:
            template = "{}={},"
        lines = [f'&{name}']
        for key, value in content.items():
            if value is None:
                continue
            lines.append(template.format(key, value))
        lines.append('/')
        return sep.join(lines)
    
    def model_dump_mdin(self):
        content = self.model_dump()
        content.update(self.extra)
        return self.write_namelist(
            self.name,
            content,
            self.sep
        )
    

def check_mask(mask: str) -> str:
    if mask == '':
        return "''"
    else:
        if not mask.startswith("'"):
            mask = "'" + mask
        if not mask.endswith("'"):
            mask += "'"
        return mask
Mask = Annotated[str, AfterValidator(check_mask)]


class AmberCntrlSettings(AmberNamelist):
    name: str = Field(default='cntrl', init=False, exclude=True)
    sep: str = Field(default='\n', init=False, exclude=True)
    # Energy minimization
    imin: int = 0  # 0 - no minimization; 1 - minimization
    ntmin: int = 2 # 0 - CG; 1 - `ncyc` steps SD then CG; 2 - SD
    ncyc: int = 10
    dx0: float = 0.05
    # Simulation settings
    ntx: int = 1   # 1 - start from scratch, 5 - start from a restart file
    irest: int = 0 # 0 - start from scratch; 1 - restart simulation
    nstlim: int = Field(default=10, validation_alias='num_steps') # number of steps
    dt: float = 0.001 # time step in ps
    # Output control
    ofreq: int = Field(default=None, exclude=True)
    efreq: int = Field(default=None, exclude=True)
    ntpr: int = None # freq to dump energies in mdout
    ntwe: int = None # freq to dump energies in mden
    ntwr: int = None # freq to write restart file
    ntwx: int = None # freq to write trajectory file
    iwrap: int = 0   # 1 - process PBC when dump trajectories; 0 - don't process PBC
    # Restraint
    ntr: int = 0
    restraintmask: Mask = "!:WAT,Cl-,K+,Na+,NA,CL & !@H="
    restraint_wt: float = 5.0
    # SHAKE constraint
    ntc: int = 2
    ntf: int = 1
    # Others
    cut: float = 10.0  # non-bonded cutoff, in angstrom
    nmropt: int = 0  # Varying conditions, useful in heating
    # Heat/Pressure coupling
    ntb: int = 1  # 0 - non-preodic; 1 - constant volume; 2 - constant pressure
    ntp: int = 0  # 0 - no pressure coupling; 1 - isotropic coupling; 2 - anisotropic coupling
    ntt: int = 3  # 0 - NVE; 1 - weak coupling; 2 - Andersen-like coupling; 3 - Langevin dynamics
    gamma_ln: float = 2.0 # collision freq, in ps^-1
    temp0: float = 298.15  # temperature to be kept
    tempi: float = 0.0  # initial temperature 
    barostat: int = 2  # 1 - Berenden; 2 - MC
    pres0: float = 1.0 # pressure to be kept in bar
    taup: float = 5.0  # pressure relaxation time

    model_config = {
        "populate_by_name": True  # Allow population by both alias and field name
    }

    @model_validator(mode='after')
    def validate_output_freqs(self) -> Self:
        if self.ofreq is None:
            attrs = ['ntwr', 'ntwx']
            if self.efreq is None:
                attrs.append('ntpr')
                attrs.append('ntwe')
            for attr in attrs:
                assert getattr(self, attr) is not None, f'"{attr}" or "ofreq" must be set'
        self.efreq = self.ofreq if self.efreq is None else self.efreq
        self.ntpr = self.efreq if self.ntpr is None else self.ntpr
        self.ntwe = self.efreq if self.ntwe is None else self.ntwe
        self.ntwr = self.ofreq if self.ntwr is None else self.ntwr
        self.ntwx = self.ofreq if self.ntwx is None else self.ntwx
        return self


class AmberFreeEnergyCntrlSettings(AmberCntrlSettings):
    # Free Energy
    ifsc: int = 0
    icfe: int = 0
    tishake: int = 0
    clambda: float = 0.0
    lambdas: list[float] = Field(default_factory=list, exclude=True)
    ifmbar: int = 1
    bar_intervall: int = None
    mbar_states: int = Field(default=0, init=False)
    mbar_lambda: str = Field(default="", init=False)
    timask1: Mask = "" 
    timask2: Mask = ""
    scmask1: Mask = "" 
    scmask2: Mask = "" 
    scalpha: float = 0.5 
    scbeta: float = 1.0
    gti_cut: int = 1
    gti_output: int = 0
    gti_add_sc: int = 6
    gti_scale_beta: int = 1
    gti_cut_sc_on: float = None
    gti_cut_sc_off: float = None
    gti_lam_sch: int = 1
    gti_ele_sc: int = 1
    gti_vdw_sc: int = 1
    gti_cut_sc: int = 2
    gti_ele_exp: int = 2
    gti_vdw_exp: int = 2
    numexchg: int = 0
    gremd_acyc: int = None

    @model_validator(mode='after')
    def validate_gti_cut_on_off(self) -> Self:
        self.gti_cut_sc_off = self.cut if self.gti_cut_sc_off is None else self.gti_cut_sc_off
        self.gti_cut_sc_on = self.gti_cut_sc_off - 2.0 if self.gti_cut_sc_on is None else self.gti_cut_sc_on
        assert self.gti_cut_sc_on < self.gti_cut_sc_off, "gti_cut_sc_on must be smaller than gti_cut_sc_off"
        return self
    
    @model_validator(mode='after')
    def validate_mbar(self) -> Self:
        self.bar_intervall = self.efreq if self.bar_intervall is None else self.bar_intervall
        self.mbar_states = len(self.lambdas)
        self.mbar_lambda = ','.join(str(x) for x in self.lambdas)
        self.gremd_acyc = len(self.lambdas) % 2 if self.gremd_acyc is None else self.gremd_acyc
        assert self.gremd_acyc == 0 or self.gremd_acyc == 1
        return self


class AmberWtSettings(AmberNamelist):
    name: str = Field(default='wt', init=False, exclude=True)
    sep: str = Field(default=' ', init=False, exclude=True)
    type: str
    istep1: int = None
    istep2: int = None
    value1: int = None
    value2: int = None

class AmberRstSettings(AmberNamelist):
    name: str = Field(default='rst', init=False, exclude=True)
    sep: str = Field(default=' ', init=False, exclude=True)
    iat: list[int]
    r1: float
    r2: float
    r3: float
    r4: float
    rk2: float
    rk3: float

    @field_serializer('iat', when_used='always')
    def serialize_iat(self, iat: list[int]):
        return ','.join(str(x) for x in iat)


class AmberMdin(BaseModel):
    cntrl: AmberCntrlSettings
    wt: list[AmberWtSettings] = Field(default_factory=list)
    rst: list[AmberRstSettings] = Field(default_factory=list)

    def model_dump_mdin(self):
        lines = [self.cntrl.model_dump_mdin()]
        if len(self.wt) > 0 and self.wt[-1].type != 'END':
            self.wt.append(AmberWtSettings(type='END'))
        for item in self.wt:
            lines.append(item.model_dump_mdin())
        for item in self.rst:
            lines.append(item.model_dump_mdin())
        return '\n'.join(lines)
    

if __name__ == '__main__':
    # settings = AmberWtSettings(type='END')
    # settings = AmberRstSettings(iat=[1,2], r1=100.0, r2=100.0, r3=100.0, r4=100.0, rk2=100.0, rk3=100.0)
    # settings = AmberCntrlSettings(ofreq=100)
    settings = AmberMdin(
        cntrl=AmberCntrlSettings(ofreq=100),
        wt=[AmberWtSettings(type='END')]
    )
    print(settings.model_dump_mdin())