"""Top-level configuration model for ABFE setup and the ABFE pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..config.amber.simulation import AmberFepSimulationConfig, AmberSimulationConfig
from ..boresch.config import BoreschRestraintGeneratorConfig
from ..smff.config import LigandParamConfig


class AmberAbfeConfig(BaseModel):
    """Configuration for ABFE setup (``easybfe abfe setup``) and the full ABFE
    pipeline (``easybfe abfe pipeline`` / :class:`easybfe.abfe.piepline.ABFE`).

    The ``ligand`` / ``protein`` / ``output_dir`` inputs may be supplied here or
    overridden through the CLI/Python arguments. The pipeline additionally uses
    :attr:`ligand_param` (to parameterize a raw ligand) and :attr:`boresch_md`
    (the plain protein-ligand MD run only when :attr:`boresch` selects a
    trajectory-based finder).
    """

    protein: Optional[Path] = Field(default=None, description="Protein PDB path.")
    ligand: Optional[Path] = Field(
        default=None,
        description="Ligand input: a parameterized ligand directory or a raw ligand file (e.g. SDF).",
    )
    ligand_batch: Optional[list[Path]] = Field(
        default=None,
        description="List of ligand directories for batch setup (mutually exclusive with ligand).",
    )
    output_dir: Optional[Path] = Field(
        default=None, description="Output directory for a single run."
    )
    ligand_base: Optional[Path] = Field(
        default=None,
        description="If set, ligand paths are resolved as ligand_base / relative_path.",
    )
    output_base: Optional[Path] = Field(
        default=None,
        description="Parent directory for per-ligand run outputs (required for batch setup).",
    )
    ligand_param: LigandParamConfig = Field(
        default_factory=LigandParamConfig,
        description="Ligand parameterization settings used when a raw ligand file is provided.",
    )
    boresch: BoreschRestraintGeneratorConfig = Field(
        default_factory=BoreschRestraintGeneratorConfig,
        description="Boresch restraint placement settings for the complex/restraint legs.",
    )
    boresch_md: AmberSimulationConfig = Field(
        default_factory=AmberSimulationConfig,
        description="Plain protein-ligand MD used only when the Boresch algorithm needs a trajectory.",
    )
    complex: AmberFepSimulationConfig = Field(
        default_factory=AmberFepSimulationConfig,
        description="FEP simulation settings for the complex leg.",
    )
    solvent: AmberFepSimulationConfig = Field(
        default_factory=AmberFepSimulationConfig,
        description="FEP simulation settings for the solvent leg.",
    )
    restraint: AmberFepSimulationConfig = Field(
        default_factory=AmberFepSimulationConfig,
        description="FEP simulation settings for the restraint leg.",
    )
    early_stop_threshold: Optional[float] = Field(
        default=None,
        description=(
            "If set, run only the pre-production stages of every leg first and "
            "estimate the binding free energy from the second-to-last workflow "
            "stage. When that estimate (kcal/mol) is greater than this threshold "
            "the ligand is treated as a weak/non-binder and the final production "
            "stage is skipped to save compute. ``None`` disables early stopping "
            "and the full workflow (including production) is always run."
        ),
    )
