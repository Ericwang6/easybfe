from __future__ import annotations
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, field_validator, field_serializer
from .simulation import AmberFepSimulationConfig


class BoreschRestraintGeneratorConfig(BaseModel):
    algorithm: str = 'rxrx'
    rst_wts: tuple[float, float, float, float, float, float] = Field(default=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0))
    options: dict[str, Any] = Field(default_factory=dict)



class AmberAbfeConfig(BaseModel):
    protein: Optional[Path] = None
    ligand: Optional[Path] = None
    ligand_batch: Optional[list[Path]] = None
    output_dir: Optional[Path] = None
    boresch: BoreschRestraintGeneratorConfig
    complex: AmberFepSimulationConfig
    solvent: AmberFepSimulationConfig
    restraint: AmberFepSimulationConfig

    


