from __future__ import annotations
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
    override.update(cfg.model_dump(exclude_unset=True))
    return cfg.__class__.model_validate(override)


class AmberStepConfig(BaseModel):
    type: Literal['em', 'heat', 'pres', 'prod', 'prod_nvt'] = 'prod'
    name: str
    exec: Literal['pmemd', 'pmemd.cuda', 'pmemd.cuda.MPI', 'pmemd.MPI'] = 'pmemd.cuda'
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


def default_workflow():
    return [
        AmberStepConfig.model_validate({"type": "em", "name": "01.em"}),
        AmberStepConfig.model_validate({"type": "heat", "name": "02.heat"}),
        AmberStepConfig.model_validate({"type": "pres", "name": "03.pres"}),
        AmberStepConfig.model_validate({"type": "prod", "name": "04.pre_prod"}),
        AmberStepConfig.model_validate({"type": "prod", "name": "05.prod"})
    ]


class AmberSimulationConfig(SetupConfig):

    do_hmr_water: bool = False
    workflow: list[AmberStepConfig] = None

    @model_validator(mode='after')
    def set_workflow(self):
        if self.workflow is None:
            self.workflow = default_workflow()
        for i in range(1, len(self.workflow)):
            if self.workflow[i-1].type != 'em':
                self.workflow[i].cntrl = _set_defaults(self.workflow[i].cntrl, {'ntx': 5, 'irest': 1})
        return self
    

class AmberFepSimulationConfig(AmberSimulationConfig):

    use_charge_change: bool = True 
    use_settle_for_alchemical_water: bool = True
    lambdas: list[float] | None = None
    num_lambdas: int = 16

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
            if step.type != 'em':
                step.exec = step.exec + '.MPI' if not step.exec.endswith('.MPI') else step.exec
        return self