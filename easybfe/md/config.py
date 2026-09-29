"""Configuration model for the one-line plain MD pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..config.analysis import PlainMDAnalysisConfig
from ..config.amber.simulation import AmberSimulationConfig
from ..smff.config import LigandParamConfig


class MDPipelineConfig(BaseModel):
    """Configuration for ``easybfe md pipeline`` / :class:`easybfe.md.pipeline.MD`.

    Any of ``protein`` / ``ligand`` / ``output_dir`` may be supplied here or
    overridden through the CLI/Python arguments. Which of ``protein`` and
    ``ligand`` are given decides the system: protein only, ligand only (in
    solvent), or protein-ligand complex.
    """

    protein: Optional[Path] = Field(default=None, description="Protein PDB path.")
    ligand: Optional[Path] = Field(
        default=None,
        description="Ligand input: a parameterized ligand directory, a .ligpack archive, or a raw ligand file (e.g. SDF).",
    )
    output_dir: Optional[Path] = Field(default=None, description="Output directory for the run.")
    task_name: str = Field(
        default="",
        description="Name shown on analysis plots. Defaults to the output directory name.",
    )
    ligand_param: LigandParamConfig = Field(
        default_factory=LigandParamConfig,
        description="Ligand parameterization settings used when a raw ligand file is provided.",
    )
    simulation: AmberSimulationConfig = Field(
        default_factory=AmberSimulationConfig,
        description="System building and MD workflow settings.",
    )
    analysis: PlainMDAnalysisConfig = Field(
        default_factory=PlainMDAnalysisConfig,
        description=(
            "Trajectory analysis settings. Fields left unset get defaults for the system type "
            "(selections; interaction analysis and GBSA only for a protein-ligand complex)."
        ),
    )
