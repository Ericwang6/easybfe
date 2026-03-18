from __future__ import annotations
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, field_validator, field_serializer
from .simulation import AmberFepSimulationConfig


class AtomMappingConfig(BaseModel):
    algorithm: str = 'lomap'
    options: dict[str, Any] = Field(default_factory=dict)


class AmberLigandRbfeConfig(BaseModel):
    protein: Optional[Path] = None
    ligandA: Optional[Path] = None
    ligandB: Optional[Path] = None
    output_dir: Optional[Path] = None
    atom_mapping: AtomMappingConfig = Field(default_factory=AtomMappingConfig)
    complex: AmberFepSimulationConfig = Field(default_factory=AmberFepSimulationConfig)
    solvent: AmberFepSimulationConfig = Field(default_factory=AmberFepSimulationConfig)
    gas: Optional[AmberFepSimulationConfig] = None

    


