import os
from pathlib import Path

import numpy as np
import pytest

from easybfe.core.protein import Protein
from easybfe.core.ligand import Ligand
from easybfe.amber.boresch import (
    BORESCH_FINDER_REGISTRY,
    BoreschRestraint,
    RxRxBoreschRestraintsFinder,
    UserSpecifiedBoreschRestraint,
)


def _load_test_protein_and_ligand():
    """Helper to load tyk2 protein and jmc_23 ligand from test data."""
    base_dir = Path(__file__).parent
    protein_pdb = base_dir / "data" / "tyk2_pdbfixer.pdb"
    ligand_sdf = base_dir / "_test_ligand_abfe_old" / "abfe_output" / "complex" / "jmc_23.sdf"

    assert protein_pdb.exists(), f"Missing test protein file: {protein_pdb}"
    assert ligand_sdf.exists(), f"Missing test ligand file: {ligand_sdf}"

    protein = Protein.from_pdb(protein_pdb, name="tyk2")
    ligand = Ligand.from_file(ligand_sdf, only_first=True, use_stem_as_name=True)
    return protein, ligand


def test_rxrx_boresch_restraints_finder_runs_and_returns_restraint():
    """Ensure RxRxBoreschRestraintsFinder.find runs and returns a valid BoreschRestraint."""
    protein, ligand = _load_test_protein_and_ligand()

    finder = RxRxBoreschRestraintsFinder(protein, ligand)
    restr = finder.find()

    print(restr)

    # Basic type and attribute checks
    assert isinstance(restr, BoreschRestraint)
    assert isinstance(restr.protein_anchors, tuple)
    assert isinstance(restr.ligand_anchors, tuple)
    assert len(restr.protein_anchors) == 3
    assert len(restr.ligand_anchors) == 3

    # After computing restraint values, rst_vals should be six floats
    protein_pdb = protein.to_openmm()
    protein_pos = np.array([[v.x, v.y, v.z] for v in protein_pdb.positions]) * 10
    ligand_mol = ligand.get_rdmol()
    ligand_pos = ligand_mol.GetConformer().GetPositions()

    vals = restr.compute_rst_vals(protein_pos, ligand_pos)
    assert isinstance(vals, tuple)
    assert len(vals) == 6
    for v in vals:
        assert isinstance(v, float)


def test_list_boresch_finders():
    """Registry contains expected finder names."""
    names = BORESCH_FINDER_REGISTRY.names()
    assert isinstance(names, list)
    assert "rxrx" in names
    assert "user" in names
    assert names == sorted(names)


def test_get_boresch_finder_returns_correct_classes():
    """BORESCH_FINDER_REGISTRY.get returns the registered finder class for each name."""
    assert BORESCH_FINDER_REGISTRY.get("rxrx") is RxRxBoreschRestraintsFinder
    assert BORESCH_FINDER_REGISTRY.get("user") is UserSpecifiedBoreschRestraint


def test_get_boresch_finder_unknown_raises():
    """BORESCH_FINDER_REGISTRY.get raises KeyError with available names for unknown key."""
    with pytest.raises(KeyError) as exc_info:
        BORESCH_FINDER_REGISTRY.get("nonexistent")
    assert "nonexistent" in str(exc_info.value)
    assert "rxrx" in str(exc_info.value)
    assert "user" in str(exc_info.value)


def test_finder_via_registry_works():
    """A finder obtained via BORESCH_FINDER_REGISTRY.create can be used."""
    protein, ligand = _load_test_protein_and_ligand()
    finder = BORESCH_FINDER_REGISTRY.create("rxrx", protein, ligand)
    restr = finder.find()
    assert isinstance(restr, BoreschRestraint)
    assert len(restr.protein_anchors) == 3
    assert len(restr.ligand_anchors) == 3


def test_user_specified_boresch_restraint():
    """UserSpecifiedBoreschRestraint returns a restraint with the given anchors and computed rst_vals."""
    protein, ligand = _load_test_protein_and_ligand()

    # Get valid anchor indices from RxRx so we know they are in range
    rxrx_finder = RxRxBoreschRestraintsFinder(protein, ligand)
    rxrx_restr = rxrx_finder.find()
    protein_anchors = rxrx_restr.protein_anchors
    ligand_anchors = rxrx_restr.ligand_anchors

    finder = UserSpecifiedBoreschRestraint(
        protein, ligand,
        protein_anchors=protein_anchors,
        ligand_anchors=ligand_anchors,
    )
    restr = finder.find()

    assert isinstance(restr, BoreschRestraint)
    assert restr.protein_anchors == protein_anchors
    assert restr.ligand_anchors == ligand_anchors
    assert len(restr.rst_vals) == 6
    for v in restr.rst_vals:
        assert isinstance(v, float)
    # Same anchors and positions => same restraint values as RxRx
    assert restr.rst_vals == rxrx_restr.rst_vals

