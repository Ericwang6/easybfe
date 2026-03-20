from typing import Any, Dict, List, Optional, Any

from pydantic import BaseModel, Field, model_validator


class ProteinPrepareConfig(BaseModel):
    """
    Configuration for running :meth:`easybfe.protein_prep.fixer.ProteinFixer.run`.

    This model groups the keyword arguments used to control the protein
    preparation and fixing workflow. It is intended to be a serializable
    representation of the options passed to
    :meth:`easybfe.protein_prep.fixer.ProteinFixer.run`.
    """

    keep_chains: Optional[List[str]] = Field(
        None,
        description=(
            "Chain IDs of polymer chains to keep. If None, keep all polymer chains."
        ),
        json_schema_extra={"is_cli": True}
    )
    keep_water: bool = Field(
        True,
        description="If True, retain water residues.",
        json_schema_extra={"is_cli": True}
    )
    keep_ions: bool = Field(
        True,
        description="If True, retain ion residues.",
        json_schema_extra={"is_cli": True}
    )
    extra_keep: Optional[List[str]] = Field(
        None,
        description="Additional heterogen residue names to retain.",
        json_schema_extra={"is_cli": True}
    )
    water_ion_distance: Optional[float] = Field(
        None,
        description=(
            "If set, only keep water/ions within this distance (Angstrom) from "
            "kept residues. Overrides keep_water and keep_ions."
        ),
        json_schema_extra={"is_cli": True}
    )
    replace_nonstandard: bool = Field(
        True,
        description=(
            "If True, replace all non-standard residues with their standard "
            "equivalents."
        ),
        json_schema_extra={"is_cli": True}
    )
    skip_missing_terminal_residues: bool = Field(
        False,
        description=(
            "If True, skip adding missing residues at chain termini when finding "
            "and fixing missing residues."
        ),
        json_schema_extra={"is_cli": True}
    )
    max_num_consecutive_missing_residues: Optional[int] = Field(
        None,
        description=(
            "Maximum number of consecutive missing residues to add; longer gaps "
            "will be skipped."
        ),
        json_schema_extra={"is_cli": True}
    )
    pH: float = Field(
        7.4,
        description=(
            "pH value used to determine protonation states when adding hydrogens."
        ),
        json_schema_extra={"is_cli": True}
    )
    forcefield: Optional[Any] = Field(
        None,
        description=(
            "OpenMM ForceField object for adding hydrogens and refining positions; "
            "if None, defaults from pdbfixer/OpenMM are used."
        ),
    )
    cap_gaps: bool = Field(
        True,
        description="If True, add ACE and NME caps to skipped gaps.",
        json_schema_extra={"is_cli": True}
    )
    force_cap_terminals: bool = Field(
        False,
        description=(
            "If True, force addition of ACE at N-terminus and NME at C-terminus "
            "for all chains."
        ),
        json_schema_extra={"is_cli": True}
    )
    res_num_mapping: Optional[Dict[str, Dict[int, str]]] = Field(
        None,
        description=(
            "Mapping from chain ID to residue number mapping used to restore "
            "original PDB residue numbering and insertion codes."
        ),
    )

    @model_validator(mode='before')
    @classmethod
    def validate_extra_keep(cls, data: Dict[str, Any]):
        extra = data.get('extra_keep', list())
        if isinstance(extra, str):
            extra = extra.split(',')
        data['extra_keep'] = extra
        return data

