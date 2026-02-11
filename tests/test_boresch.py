import os
from pathlib import Path

import numpy as np

from easybfe.core.protein import Protein
from easybfe.core.ligand import Ligand
from easybfe.amber.boresch import RxRxBoreschRestraintsFinder, BoreschRestraint


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

    finder = RxRxBoreschRestraintsFinder()
    restr = finder.find(protein, ligand)

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

