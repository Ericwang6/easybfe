"""Configuration models for small-molecule force field parameterization."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LigandParamConfig(BaseModel):
    """Settings forwarded to :func:`easybfe.smff.parametrize_ligands`.

    Used by the ABFE pipeline (:class:`easybfe.abfe.piepline.ABFE`) to
    parameterize a raw ligand input (e.g. an SDF file) when an
    already-parameterized ligand directory is not supplied.
    """

    forcefield: str = Field(
        default="gaff2",
        description="Force field name or path (e.g. 'gaff2', 'openff-2.1.0', or a path to an .xml file).",
    )
    charge_method: str = Field(
        default="bcc",
        description="Partial charge assignment method (e.g. 'bcc', 'gas', 'resp').",
    )
    engine: str = Field(
        default="",
        description="Explicit engine override ('acpype', 'openff', or 'custom'). Auto-detected from forcefield when empty.",
    )
    resp_engine: str = Field(
        default="",
        description="Engine for RESP charge calculations (e.g. 'qchem'). Only used when charge_method starts with 'resp'.",
    )
