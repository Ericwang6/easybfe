from __future__ import annotations
import os, json
from pathlib import Path
from typing import Literal, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from ..analysis import PlainMDAnalysisConfig, _infer_plot_names_from_task_name, _infer_selection_from_task_type
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
            # AmberWtSettings validates type with add_quote (double quotes), so after
            # round-trip wt.type is '"TEMP0"'; check normalized value for idempotency.
            def _is_temp0(w):
                t = w.type.strip('"\'')
                return t == 'TEMP0'
            if not any(_is_temp0(wt) for wt in self.wt):
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


def default_fep_workflow():
    with open(os.path.join(os.path.dirname(__file__), 'fep_default.json')) as f:
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


class AmberPlainMDConfig(BaseModel):
    protein: Union[Path, None] = None
    ligand: Union[Path, None] = None
    output_dir: Path
    task_type: str = Field(init=False, default='')
    task_name: str = ''
    simulation: AmberSimulationConfig = Field(default_factory=AmberSimulationConfig)
    analysis: PlainMDAnalysisConfig = Field(default_factory=PlainMDAnalysisConfig)

    @field_validator('protein', 'ligand', 'output_dir', mode='before')
    @classmethod
    def _coerce_path(cls, v: Any) -> Any:
        if v is None or isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v)
        return v

    @model_validator(mode='after')
    def validate_task_type(self):
        assert not ((self.protein is None) and (self.ligand is None)), 'Protein and Ligand cannot be None at the same time!'
        if self.protein is None:
            self.task_type = 'ligand'
        elif self.ligand is None:
            self.task_type = 'protein'
        else:
            self.task_type = 'complex'
        return self
    
    @model_validator(mode='after')
    def validate_task_name(self):
        if not self.task_name:
            self.task_name = self.output_dir.name
        return self
    
    @model_validator(mode='after')
    def validate_analysis(self):
        ana = {}
        ana.update(_infer_selection_from_task_type(self.task_type))
        ana.update(_infer_plot_names_from_task_name(self.task_name))
        ana.update(self.analysis.model_dump(exclude_unset=True))
        ana = PlainMDAnalysisConfig.model_validate(ana)
        self.analysis = ana
        return self


class AmberFepSimulationConfig(AmberSimulationConfig):

    workflow: list[AmberStepConfig] = Field(default_factory=default_fep_workflow)
    use_charge_change: bool = True 
    use_settle_for_alchemical_water: bool = True
    add_restraint_for_alchem_water: bool = True
    charge_change_method: Literal['dummy_ion', 'coalchem_water'] = 'dummy_ion'
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