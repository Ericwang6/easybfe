"""Configuration models for Boresch restraint generation."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BoreschRestraintGeneratorConfig(BaseModel):
    """Settings describing how Boresch restraints are placed for the complex leg.

    The :attr:`algorithm` selects a finder registered in
    :data:`easybfe.boresch.BORESCH_FINDER_REGISTRY` (e.g. ``'rxrx'`` for the
    single-structure finder or ``'rxrx-md'`` for the trajectory-based finder).
    """

    algorithm: str = Field(
        default="rxrx",
        description="Name of the registered Boresch finder (e.g. 'rxrx', 'rxrx-md', 'user').",
    )
    rst_wts: tuple[float, float, float, float, float, float] = Field(
        default=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        description="Six Boresch force constants (bond, two angles, three dihedrals).",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to the selected finder constructor.",
    )
