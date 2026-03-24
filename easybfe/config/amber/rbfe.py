from __future__ import annotations
from typing import Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from .simulation import AmberFepSimulationConfig


class AtomMappingConfig(BaseModel):
    algorithm: str = 'lomap'
    options: dict[str, Any] = Field(default_factory=dict)


class LigandNetworkConfig(BaseModel):
    algorithm: str = "custom"
    options: dict[str, Any] = Field(default_factory=dict)


class AmberLigandRbfeConfig(BaseModel):
    protein: Optional[Path] = None
    ligandA: Optional[Path] = None
    ligandB: Optional[Path] = None
    ligand_list: Optional[List[Path]] = None
    network: LigandNetworkConfig = Field(default_factory=LigandNetworkConfig)
    output_dir: Optional[Path] = Field(
        default=None,
        description="Single-pair run output directory when output_base is not set.",
    )
    output_base: Optional[Path] = Field(
        default=None,
        description=(
            "Parent directory for run outputs. Required for network mode. "
            "Single-pair: writes to output_base / '{ligandA.name}~{ligandB.name}'."
        ),
    )
    ligand_base: Optional[Path] = Field(
        default=None,
        description=(
            "If set, ligand paths are resolved as ligand_base / relative_path. "
            "In network mode, entries in ligand_list are resolved under ligand_base."
        ),
    )
    atom_mapping: AtomMappingConfig = Field(default_factory=AtomMappingConfig)
    complex: AmberFepSimulationConfig = Field(default_factory=AmberFepSimulationConfig)
    solvent: AmberFepSimulationConfig = Field(default_factory=AmberFepSimulationConfig)
    gas: Optional[AmberFepSimulationConfig] = None

    @model_validator(mode="after")
    def validate_inputs(self) -> "AmberLigandRbfeConfig":
        has_list = self.ligand_list is not None and len(self.ligand_list) > 0
        has_pair = self.ligandA is not None or self.ligandB is not None
        if not has_list and not has_pair:
            raise ValueError("Set either ligand_list or ligandA/ligandB")
        if has_list and self.output_base is None:
            raise ValueError("output_base is required when ligand_list is set")
        if has_pair and (self.ligandA is None or self.ligandB is None):
            raise ValueError("ligandA and ligandB must both be set in single-pair mode")
        return self


