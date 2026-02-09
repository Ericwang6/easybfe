from typing import Any, Dict, List, Optional, Any

from pydantic import BaseModel, Field, model_validator


class ProteinPrepareConfig(BaseModel):
    """
    Configuration for running :meth:`easybfe.protein_prep.fixer.ProteinFixer.run`.

    This model groups the keyword arguments used to control the protein
    preparation and fixing workflow. It is intended to be a serializable
    representation of the options passed to
    :meth:`easybfe.protein_prep.fixer.ProteinFixer.run`.

    Parameters
    ----------
    output_protein : str
        Path to output PDB file where the fixed structure will be saved.
    skip_missing_terminal_residues : bool, optional
        If ``True``, skip adding missing residues at chain termini.
        See :meth:`easybfe.protein_prep.fixer.ProteinFixer.findMissingAtoms`.
    max_num_consecutive_missing_residues : int, optional
        Maximum number of consecutive missing residues to add. Gaps longer
        than this will be skipped.
    keep_water : bool, optional
        If ``True``, retain water residues when removing heterogens.
        See :meth:`easybfe.protein_prep.fixer.ProteinFixer.removeHeterogens`.
    keep_ions : bool, optional
        If ``True``, retain ion residues when removing heterogens.
        See :meth:`easybfe.protein_prep.fixer.ProteinFixer.removeHeterogens`.
    extra_keep : List[str], optional
        Additional residue names to retain when removing heterogens.
    pH : float, optional
        pH value for determining protonation states when adding hydrogens.
        See :meth:`easybfe.protein_prep.fixer.ProteinFixer.addMissingHydrogens`.
    forcefield : Any, optional
        OpenMM ``ForceField`` object for adding hydrogens and refining
        positions. If ``None``, :class:`pdbfixer.PDBFixer` defaults are used.
    cap_gaps : bool, optional
        If ``True``, add ``ACE`` and ``NME`` caps to skipped gaps.
    force_cap_terminals : bool, optional
        If ``True``, force addition of ``ACE`` at N-terminus and ``NME`` at
        C-terminus for all chains.
    res_num_mapping : Dict[str, Dict[int, str]], optional
        Mapping from chain ID to residue number mapping used to restore
        original PDB residue numbering and insertion codes.
    """

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
    keep_water: bool = Field(
        True,
        description="If True, retain water residues when removing heterogens.",
        json_schema_extra={"is_cli": True}
    )
    keep_ions: bool = Field(
        True,
        description="If True, retain ion residues when removing heterogens.",
        json_schema_extra={"is_cli": True}
    )
    extra_keep: Optional[List[str]] = Field(
        None,
        description="Additional residue names to retain when removing heterogens.",
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

