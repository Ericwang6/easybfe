import os
from pathlib import Path

import numpy as np
import pytest

from rdkit import Chem

from easybfe.core.protein import Protein
from easybfe.core.ligand import Ligand
from easybfe.boresch import (
    BORESCH_FINDER_REGISTRY,
    BoreschRestraint,
    RxRxBoreschRestraintsFinder,
    RxRxMDBoreschRestraintsFinder,
    UserSpecifiedBoreschRestraint,
    compute_bond,
    compute_angle,
    compute_dihedral,
    _bond_series,
    _angle_series,
    _dihedral_series,
    _circular_mean_deg,
    _circular_std_rad,
    _map_ligand_atom_to_candidate,
)


SELECT_REP_DIR = Path(__file__).parent / "data" / "select_rep"
SELECT_REP_TOPOLOGY = SELECT_REP_DIR / "prod_processed.pdb"
SELECT_REP_TRAJECTORY = SELECT_REP_DIR / "prod_processed.xtc"
SELECT_REP_LIGAND = SELECT_REP_DIR / "best_pose.sdf"


def _load_test_protein_and_ligand():
    """Helper to load tyk2 protein and jmc_23 ligand from test data."""
    base_dir = Path(__file__).parent
    protein_pdb = base_dir / "data" / "tyk2_pdbfixer.pdb"
    ligand_sdf = base_dir / "data" / "jmc_23.sdf"

    assert protein_pdb.exists(), f"Missing test protein file: {protein_pdb}"
    assert ligand_sdf.exists(), f"Missing test ligand file: {ligand_sdf}"

    protein = Protein.from_pdb(protein_pdb, name="tyk2")
    ligand = Ligand.from_file(ligand_sdf, only_first=True, name_from_stem=True)
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


# ---------------------------------------------------------------------------
# RxRxMDBoreschRestraintsFinder
# ---------------------------------------------------------------------------


def _build_complex_protein_and_ligand(tmp_path: Path):
    """Build a protein-only ``Protein`` matching the select_rep trajectory's
    ``protein`` selection, plus the ligand from ``best_pose.sdf``."""
    import MDAnalysis as mda

    universe = mda.Universe(str(SELECT_REP_TOPOLOGY))
    protein_atoms = universe.select_atoms("protein")
    protein_pdb = tmp_path / "protein_only.pdb"
    protein_atoms.write(str(protein_pdb))
    protein = Protein.from_pdb(protein_pdb, name="select_rep_protein")
    ligand = Ligand.from_file(SELECT_REP_LIGAND, only_first=True, name_from_stem=True)
    return protein, ligand


def test_rxrx_md_registered():
    """Registry exposes 'rxrx-md' and maps to the new finder class."""
    assert "rxrx-md" in BORESCH_FINDER_REGISTRY.names()
    assert BORESCH_FINDER_REGISTRY.get("rxrx-md") is RxRxMDBoreschRestraintsFinder


def test_vectorized_series_match_scalar():
    """Vectorized DOF series match the scalar reference implementations."""
    rng = np.random.default_rng(0)
    n_frames = 8
    p0 = rng.normal(size=(n_frames, 3))
    p1 = rng.normal(size=(n_frames, 3))
    p2 = rng.normal(size=(n_frames, 3))
    p3 = rng.normal(size=(n_frames, 3))

    bonds = _bond_series(p0, p1)
    angles = _angle_series(p0, p1, p2)
    dihedrals = _dihedral_series(p0, p1, p2, p3)

    for f in range(n_frames):
        assert abs(bonds[f] - compute_bond(p0[f], p1[f])) < 1e-9
        assert abs(angles[f] - compute_angle(p0[f], p1[f], p2[f])) < 1e-7
        assert abs(dihedrals[f] - compute_dihedral(p0[f], p1[f], p2[f], p3[f])) < 1e-7


def test_circular_stats():
    """Circular mean wraps around 0 and circular std is zero for constants."""
    assert abs(((_circular_mean_deg([350.0, 10.0]) + 180) % 360) - 180) < 1e-6
    assert abs(_circular_mean_deg([42.0, 42.0, 42.0]) - 42.0) < 1e-6
    assert _circular_std_rad([42.0, 42.0, 42.0]) < 1e-6
    assert _circular_std_rad([0.0, 90.0, 180.0, 270.0]) > 0.5


def test_map_ligand_atom_to_candidate():
    """Terminal interacting atoms map to a bonded nonterminal heavy atom."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))  # acetic acid
    # Identify atoms by element / connectivity.
    carboxyl_c = None
    carbonyl_o = None
    hydroxyl_o = None
    methyl_c = None
    for atom in mol.GetAtoms():
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if atom.GetSymbol() == "C" and any(n.GetSymbol() == "O" for n in heavy):
            carboxyl_c = atom.GetIdx()
        elif atom.GetSymbol() == "C":
            methyl_c = atom.GetIdx()
        elif atom.GetSymbol() == "O":
            if len(heavy) == 1:
                carbonyl_o = atom.GetIdx()
            else:
                hydroxyl_o = atom.GetIdx()

    # The carboxyl carbon has two heavy neighbors -> maps to itself.
    assert _map_ligand_atom_to_candidate(mol, carboxyl_c) == carboxyl_c
    # Terminal carbonyl oxygen -> reassigned to the carboxyl carbon.
    assert _map_ligand_atom_to_candidate(mol, carbonyl_o) == carboxyl_c
    # Methyl carbon (single heavy neighbor) -> reassigned to the carboxyl carbon.
    assert _map_ligand_atom_to_candidate(mol, methyl_c) == carboxyl_c


def test_find_requires_trajectory():
    """The finder only handles the trajectory case and errors without one."""
    protein, ligand = _load_test_protein_and_ligand()
    finder = RxRxMDBoreschRestraintsFinder(protein, ligand)
    with pytest.raises(ValueError, match="topology"):
        finder.find()


def test_trajectory_loader_shapes(tmp_path):
    """The MDAnalysis loader splits the complex trajectory into protein/ligand."""
    protein, ligand = _build_complex_protein_and_ligand(tmp_path)
    finder = RxRxMDBoreschRestraintsFinder(
        protein,
        ligand,
        topology=SELECT_REP_TOPOLOGY,
        trajectory=SELECT_REP_TRAJECTORY,
        ligand_residue_name="MOL",
    )
    protein_frames, ligand_frames = finder._load_trajectory_frames()

    n_ligand = ligand.get_rdmol().GetNumAtoms()
    n_protein = len(list(protein.to_openmm().topology.atoms()))

    assert protein_frames.ndim == 3 and protein_frames.shape[2] == 3
    assert ligand_frames.ndim == 3 and ligand_frames.shape[2] == 3
    assert protein_frames.shape[0] == ligand_frames.shape[0]
    assert ligand_frames.shape[1] == n_ligand
    assert protein_frames.shape[1] == n_protein


def test_index_maps(tmp_path):
    """PLIP serial -> ligand/protein local index maps are consistent."""
    protein, ligand = _build_complex_protein_and_ligand(tmp_path)
    finder = RxRxMDBoreschRestraintsFinder(
        protein,
        ligand,
        topology=SELECT_REP_TOPOLOGY,
        ligand_residue_name="MOL",
    )
    uidx_to_ligand, uidx_to_protein = finder._build_index_maps()

    n_ligand = ligand.get_rdmol().GetNumAtoms()
    n_protein = len(list(protein.to_openmm().topology.atoms()))
    assert len(uidx_to_ligand) == n_ligand
    assert len(uidx_to_protein) == n_protein
    # Local indices form a contiguous 0..N-1 range.
    assert sorted(uidx_to_ligand.values()) == list(range(n_ligand))
    assert sorted(uidx_to_protein.values()) == list(range(n_protein))


def test_interaction_candidates_from_csv(tmp_path):
    """A synthetic PLIP CSV row above threshold yields backbone candidates; a
    below-threshold row yields none."""
    import pandas as pd

    protein, ligand = _build_complex_protein_and_ligand(tmp_path)
    finder = RxRxMDBoreschRestraintsFinder(
        protein,
        ligand,
        topology=SELECT_REP_TOPOLOGY,
        ligand_residue_name="MOL",
        occupancy_threshold=0.5,
    )

    protein_atoms = list(protein.to_openmm().topology.atoms())
    ligand_mol = ligand.get_rdmol()
    uidx_to_ligand, uidx_to_protein = finder._build_index_maps()
    ligand_to_uidx = {local: uidx for uidx, local in uidx_to_ligand.items()}
    protein_to_uidx = {local: uidx for uidx, local in uidx_to_protein.items()}

    # Pick a ligand heavy atom that is a valid candidate anchor.
    ligand_local = None
    for local in range(ligand_mol.GetNumAtoms()):
        if _map_ligand_atom_to_candidate(ligand_mol, local) is not None:
            ligand_local = local
            break
    assert ligand_local is not None
    ligand_serial = ligand_to_uidx[ligand_local] + 1

    # Pick a protein backbone atom whose residue has a full CA/C/N backbone.
    protein_local = None
    for local in range(len(protein_atoms)):
        residue = protein_atoms[local].residue
        names = {atom.name for atom in residue.atoms()}
        if {"CA", "C", "N"}.issubset(names):
            protein_local = local
            break
    assert protein_local is not None
    protein_serial = protein_to_uidx[protein_local] + 1

    columns = [
        "interaction", "resname", "resnr", "chain", "ligand_idx",
        "protein_idx", "water_idx", "dist", "ratio", "residue_ratio",
    ]

    df_high = pd.DataFrame(
        [{
            "interaction": "hydrogen_bond", "resname": "ALA", "resnr": "1", "chain": "A",
            "ligand_idx": str(ligand_serial), "protein_idx": str(protein_serial),
            "water_idx": None, "dist": 3.0, "ratio": 0.9, "residue_ratio": 0.9,
        }],
        columns=columns,
    )
    candidates_high = finder._interaction_candidates(df_high, ligand_mol, protein_atoms)
    assert len(candidates_high) > 0

    df_low = df_high.copy()
    df_low.loc[0, "ratio"] = 0.1
    candidates_low = finder._interaction_candidates(df_low, ligand_mol, protein_atoms)
    assert len(candidates_low) == 0


def test_full_find_with_plip(tmp_path):
    """End-to-end find() over the select_rep trajectory, generating PLIP CSV."""
    pytest.importorskip("plip")
    protein, ligand = _build_complex_protein_and_ligand(tmp_path)
    finder = RxRxMDBoreschRestraintsFinder(
        protein,
        ligand,
        topology=SELECT_REP_TOPOLOGY,
        trajectory=SELECT_REP_TRAJECTORY,
        ligand_residue_name="MOL",
        use_mpi=False,
    )
    restr = finder.find()

    assert isinstance(restr, BoreschRestraint)
    assert len(restr.protein_anchors) == 3
    assert len(restr.ligand_anchors) == 3
    assert len(restr.rst_vals) == 6
    assert 45.0 <= restr.rst_vals[1] <= 135.0
    assert 45.0 <= restr.rst_vals[2] <= 135.0

