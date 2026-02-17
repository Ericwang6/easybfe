from __future__ import annotations
import os, json
from typing import Literal, Any
from pydantic import BaseModel, Field, model_validator
from .basic import AmberCntrlSettings, AmberWtSettings, AmberRstSettings, create_default_setting
from ..setup import SetupConfig

__all__ = [
    'AmberStepConfig',
    'AmberSimulationConfig',
    'AmberFepSimulationConfig',
    'default_workflow',
]


def _set_defaults(cfg: BaseModel, override: dict[str, Any]):
    cfg_dict = cfg.model_dump()
    cfg_dict.update(override)
    return cfg.__class__.model_validate(cfg_dict)


class AmberStepConfig(BaseModel):
    type: Literal['em', 'heat', 'pres', 'prod', 'prod_nvt'] = 'prod'
    name: str
    exec: Literal['pmemd', 'pmemd.cuda', 'pmemd.cuda.MPI', 'pmemd.MPI'] = 'pmemd.cuda'
    use_remd: bool = True
    use_mpi: bool = True
    cntrl: AmberCntrlSettings = Field(default_factory=AmberCntrlSettings)
    wt: list[AmberWtSettings] = Field(default_factory=list)
    rst: list[AmberRstSettings] = Field(default_factory=list)

    @model_validator(mode='after')
    def set_type_defaults(self):
        if self.type == 'em':
            self.cntrl = _set_defaults(self.cntrl, create_default_setting(em=True, nvt=True, restraint=False)['cntrl'])
        elif self.type == 'heat':
            override = create_default_setting(em=False, nvt=True, restraint=True)['cntrl']
            override['nmropt'] = 1
            self.cntrl = _set_defaults(self.cntrl, override)
            if not any([wt.type == 'TEMP0' for wt in self.wt]):
                self.wt.append(
                    AmberWtSettings(
                        type='TEMP0',
                        istep1=1,
                        istep2=self.cntrl.nstlim // 2,
                        value1=self.cntrl.tempi,
                        value2=self.cntrl.temp0
                    )
                )                
        elif self.type == 'pres':
            self.cntrl = _set_defaults(self.cntrl, create_default_setting(em=False, nvt=False, restraint=True)['cntrl'])
        elif self.type == 'prod_nvt':
            self.cntrl = _set_defaults(self.cntrl, create_default_setting(em=False, nvt=True, restraint=False)['cntrl'])
        else:
            self.cntrl = _set_defaults(self.cntrl, create_default_setting(em=False, nvt=False, restraint=False)['cntrl'])
        return self


def default_md_workflow():
    with open(os.path.join(os.path.dirname(__file__), 'md_default.json')) as f:
        jdatas = json.load(f)
    return [AmberStepConfig.model_validate(jdatas[i]) for i in range(len(jdatas))]


def default_abfe_workflow():
    with open(os.path.join(os.path.dirname(__file__), 'abfe_default.json')) as f:
        jdatas = json.load(f)
    return [AmberStepConfig.model_validate(jdatas[i]) for i in range(len(jdatas))]


class AmberSimulationConfig(SetupConfig):

    do_hmr_water: bool = False
    workflow: list[AmberStepConfig] = Field(default_factory=default_md_workflow)

    @model_validator(mode='after')
    def set_workflow(self):
        for i in range(1, len(self.workflow)):
            if self.workflow[i-1].type != 'em':
                self.workflow[i].cntrl = _set_defaults(self.workflow[i].cntrl, {'ntx': 5, 'irest': 1})
        return self
    

class AmberFepSimulationConfig(AmberSimulationConfig):

    use_charge_change: bool = True 
    use_settle_for_alchemical_water: bool = True
    lambdas: list[float] | None = None
    num_lambdas: int = 16
    reduce_storage: bool = True

    @model_validator(mode='after')
    def validate_nproc(self):
        if self.num_procs <= 0:
            self.num_procs = self.num_lambdas
        else:
            self.num_procs = self.num_lambdas * max(1, self.num_procs // self.num_lambdas)
        return self

    @model_validator(mode='after')
    def validate_lambdas(self):
        if self.lambdas is None:
            self.lambdas = [i/(self.num_lambdas-1) for i in range(self.num_lambdas)]
        else:
            self.num_lambdas = len(self.lambdas)
        return self

    @model_validator(mode='after')
    def set_free_energy(self):
        for step in self.workflow:
            step.cntrl = _set_defaults(step.cntrl, {'ifsc': 1, 'icfe': 1, 'ntf': 1, 'ifmbar': 1, 'mbar_lambda': self.lambdas})
        return self