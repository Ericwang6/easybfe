from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator, field_validator, field_serializer

__all__ = [
    'AmberNamelist',
    'AmberCntrlSettings',
    'AmberWtSettings',
    'AmberRstSettings',
    'AmberMdin',
    'create_default_setting',
]


class AmberNamelist(BaseModel):
    name: str
    sep: str = "\n"
    extra: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @staticmethod
    def write_namelist(name: str, content: dict[str, Any], sep: str = '\n'):
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
    

def add_quote(inp: str, single=True) -> str:
    quote = "'" if single else '"'
    before = quote if not inp.startswith(quote) else ""
    after = quote if not inp.endswith(quote) else ""
    return f'{before}{inp}{after}'


class AmberCntrlSettings(AmberNamelist):
    name: str = Field(default='cntrl', init=False, exclude=True)
    sep: str = Field(default='\n', init=False, exclude=True)
    # Energy minimization
    imin: int = 0  # 0 - no minimization; 1 - minimization
    ntmin: int = 2 # 0 - CG; 1 - `ncyc` steps SD then CG; 2 - SD
    ncyc: int = 10
    maxcyc: int = 2000
    dx0: float = Field(default=0.05, validation_alias='step_size')
    # Simulation settings
    ntx: int = 1   # 1 - start from scratch, 5 - start from a restart file
    irest: int = 0 # 0 - start from scratch; 1 - restart simulation
    nstlim: int = Field(default=10000, validation_alias='num_steps') # number of steps
    dt: float = 0.001 # time step in ps
    # Output control
    ofreq: int = Field(default=1000, exclude=True)
    efreq: Optional[int] = Field(default=None, exclude=True)
    ntpr: Optional[int] = None # freq to dump energies in mdout
    ntwe: Optional[int] = None # freq to dump energies in mden
    ntwr: Optional[int] = None # freq to write restart file
    ntwx: Optional[int] = None # freq to write trajectory file
    iwrap: int = 0   # 1 - process PBC when dump trajectories; 0 - don't process PBC
    # Restraint
    ntr: int = 0
    restraintmask: str = "!:WAT,Cl-,K+,Na+,NA,CL & !@H="
    restraint_wt: float = 5.0
    # SHAKE constraint
    ntc: int = 2
    ntf: int = 2
    # Others
    cut: float = 10.0  # non-bonded cutoff, in angstrom
    nmropt: int = 0  # Varying conditions, useful in heating
    # Heat/Pressure coupling
    ntb: int = 2  # 0 - non-preodic; 1 - constant volume; 2 - constant pressure
    ntp: int = 1  # 0 - no pressure coupling; 1 - isotropic coupling; 2 - anisotropic coupling
    ntt: int = 3  # 0 - NVE; 1 - weak coupling; 2 - Andersen-like coupling; 3 - Langevin dynamics
    gamma_ln: float = 2.0 # collision freq, in ps^-1
    temp0: float = 298.15  # temperature to be kept
    tempi: float = 0.0  # initial temperature 
    barostat: int = 2  # 1 - Berenden; 2 - MC
    pres0: float = 1.01325 # pressure to be kept in bar
    taup: float = 5.0  # pressure relaxation time

    # Free Energy
    ifsc: int = 0
    icfe: int = 0
    tishake: int = 0
    clambda: float = 0.0
    ifmbar: int = 0
    bar_intervall: Optional[int] = None
    mbar_states: Optional[int] = None
    mbar_lambda: list[float] = Field(default_factory=list)
    timask1: str = "" 
    timask2: str = ""
    scmask1: str = "" 
    scmask2: str = "" 
    scalpha: float = 0.5 
    scbeta: float = 1.0
    gti_cut: int = 1
    gti_output: int = 0
    gti_add_sc: int = 6
    gti_scale_beta: int = 1
    gti_cut_sc_on: Optional[float] = None
    gti_cut_sc_off: Optional[float] = None
    gti_lam_sch: int = 1
    gti_ele_sc: int = 1
    gti_vdw_sc: int = 1
    gti_cut_sc: int = 2
    gti_ele_exp: int = 2
    gti_vdw_exp: int = 2
    numexchg: int = 0
    gremd_acyc: Optional[int] = None

    model_config = {
        "populate_by_name": True , # Allow population by both alias and field name
        "validate_default": True
    }

    @field_validator('mbar_lambda', mode='before')
    @classmethod
    def validate_list(cls, v):
        if isinstance(v, str):
            return [float(x) for x in v.split(',')]
        else:
            return v
    
    @field_serializer('mbar_lambda')
    def dump_list(self, lst: list[float]):
        return ','.join([str(x) for x in lst])
    
    @field_validator('timask1', 'timask2', 'scmask1', 'scmask2', 'restraintmask')
    @classmethod
    def validate_str_single_quote(cls, v):
        return add_quote(v, single=True)

    @model_validator(mode='after')
    def validate_output_freqs(self) -> AmberCntrlSettings:
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
    
    @model_validator(mode='after')
    def validate_fep(self) -> AmberCntrlSettings:
        if self.icfe:
            self.bar_intervall = min(self.efreq, self.nstlim) if self.bar_intervall is None else self.bar_intervall
            # We have to ensure that the BAR result is output every time after it is calculated
            self.ntpr = self.bar_intervall
            self.ntwe = self.bar_intervall

            if self.mbar_states is None:
                self.mbar_states = len(self.mbar_lambda) 
            else:
                assert self.mbar_states == len(self.mbar_lambda), 'mbar_states does not equal to length of mbar_lambda'

            self.gremd_acyc = len(self.mbar_lambda) % 2 if self.gremd_acyc is None else self.gremd_acyc
            assert self.gremd_acyc == 0 or self.gremd_acyc == 1
            
            self.gti_cut_sc_off = self.cut if self.gti_cut_sc_off is None else self.gti_cut_sc_off
            self.gti_cut_sc_on = self.gti_cut_sc_off - 2.0 if self.gti_cut_sc_on is None else self.gti_cut_sc_on
            assert self.gti_cut_sc_on < self.gti_cut_sc_off, "gti_cut_sc_on must be smaller than gti_cut_sc_off"

        return self


class AmberWtSettings(AmberNamelist):
    name: str = Field(default='wt', init=False, exclude=True)
    sep: str = Field(default=' ', init=False, exclude=True)
    type: str
    istep1: Optional[int] = None
    istep2: Optional[int] = None
    value1: float = None
    value2: float = None

    @field_validator('type')
    @classmethod
    def validate_str_double_quote(cls, v):
        return add_quote(v, single=False)


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

    @field_validator('iat', mode='before')
    @classmethod
    def validate_list(cls, v):
        if isinstance(v, str):
            return [int(x) for x in v.split(',')]
        else:
            return v
    
    @field_serializer('iat')
    def dump_list(self, v: list[int]):
        return ','.join([str(x) for x in v])


class AmberMdin(BaseModel):
    cntrl: AmberCntrlSettings
    wt: list[AmberWtSettings] = Field(default_factory=list)
    rst: list[AmberRstSettings] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_nmropt(self) -> AmberMdin:
        if len(self.rst) > 0:
            self.cntrl.nmropt = 1
        return self

    def model_dump_mdin(self):
        lines = [self.cntrl.model_dump_mdin()]
        if len(self.wt) > 0 and self.wt[-1].type != 'END':
            self.wt.append(AmberWtSettings(type='END'))
        for item in self.wt:
            lines.append(item.model_dump_mdin())
        for item in self.rst:
            lines.append(item.model_dump_mdin())
        lines.append('')
        return '\n'.join(lines)


def create_default_setting(
    em: bool = False, 
    nvt: bool = False, 
    restraint: bool = False, 
    free_energy: bool = False,
    restart: bool = False
):
    settings = {
        "cntrl": {},
        "wt": [],
        "rst": []
    }
    if em:
        settings['cntrl']['imin'] = 1
        settings['cntrl']['ntp'] = 0
        settings['cntrl']['ntb'] = 1
    if nvt:
        settings['cntrl']['ntp'] = 0
        settings['cntrl']['ntb'] = 1
    if restraint:
        settings['cntrl']['ntr'] = 1
    if restart:
        settings['cntrl']['ntx'] = 5
        settings['cntrl']['irest'] = 1
    if free_energy:
        settings['cntrl']['ifsc'] = 1
        settings['cntrl']['icfe'] = 1
        settings['cntrl']['ntf'] = 1
        settings['cntrl']['ifmbar'] = 1
    return settings


