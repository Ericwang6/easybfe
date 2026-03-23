'''
Manage all pydantic models
'''

import json
from pathlib import Path
from typing import Type, Any, Mapping, TypeVar

import yaml
from pydantic import BaseModel

from .amber.basic import (
    AmberNamelist,
    AmberCntrlSettings,
    AmberWtSettings,
    AmberRstSettings,
    AmberMdin,
    create_default_setting,
)
from .amber.simulation import (
    AmberStepConfig,
    AmberSimulationConfig,
    AmberFepSimulationConfig,
    AmberPlainMDConfig,
    default_md_workflow,
    default_fep_workflow,
)
from .protein_prep import ProteinPrepareConfig
from .analysis import PlainMDAnalysisConfig


def read_file(file_path, key=None):
    path = Path(file_path)
    suffix = path.suffix.lower()
    data = None
    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    elif suffix in (".yaml", ".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        raise NotImplementedError(
            f"Unsupported config format {path.suffix!r}; use .json, .yaml, or .yml"
        )
    return data[key] if key is not None else data
        

def load_config(cfg_cls: Type[BaseModel], config_dict: dict[str, Any], extra: dict[str, Any]):
    config_dict.update({k: v for k, v in extra.items() if v is not None})
    return cfg_cls.model_validate(config_dict)


def deep_update_dict(base: dict, update: Mapping[str, Any]) -> dict:
    for k, v in update.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            deep_update_dict(base[k], v)
        else:
            base[k] = v
    return base


M = TypeVar("M", bound=BaseModel)
def update_config(base: M, update: Mapping[str, Any]) -> M:
    data = deep_update_dict(base.model_dump(), update)
    return base.__class__.model_validate(data)