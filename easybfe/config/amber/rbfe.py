from __future__ import annotations
from typing import Any, Optional, List, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from .simulation import AmberFepSimulationConfig


class AtomMappingConfig(BaseModel):
    algorithm: str = 'lomap'
    options: dict[str, Any] = Field(default_factory=dict)


class AmberLigandRbfeConfig(BaseModel):
    protein: Optional[Path] = None
    ligandA: Optional[Path] = None
    ligandB: Optional[Path] = None
    output_dir: Optional[Path] = Field(
        default=None,
        description="Single-pair run output directory when output_base is not set.",
    )
    output_base: Optional[Path] = Field(
        default=None,
        description=(
            "Parent directory for run outputs. Required for batch (ligand_pairs). "
            "Single-pair: writes to output_base / '{ligandA.name}~{ligandB.name}'."
        ),
    )
    ligand_base: Optional[Path] = Field(
        default=None,
        description=(
            "If set, ligand paths are resolved as ligand_base / relative_path; "
            "if not set, ligandA/ligandB (or each pair entry) are full directory paths."
        ),
    )
    ligand_pairs: Optional[List[Tuple[Path, Path]]] = Field(
        default=None,
        description="If non-empty, batch mode: one RBFE job per (pair0, pair1) directory pair.",
    )
    atom_mapping: AtomMappingConfig = Field(default_factory=AtomMappingConfig)
    complex: AmberFepSimulationConfig = Field(default_factory=AmberFepSimulationConfig)
    solvent: AmberFepSimulationConfig = Field(default_factory=AmberFepSimulationConfig)
    gas: Optional[AmberFepSimulationConfig] = None

    


