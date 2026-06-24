import math
from pathlib import Path

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from rdkit import Chem

from easybfe.amber.select_rep import (
    find_rotatable_torsions,
    circular_mean,
    compute_frame_deviations,
    select_representative_frame,
)


DATA_DIR = Path(__file__).parent / "data" / "select_rep"
TOPOLOGY = DATA_DIR / "prod_processed.pdb"
TRAJECTORY = DATA_DIR / "prod_processed.xtc"
LIGAND_SDF = DATA_DIR / "best_pose.sdf"

# Persisted output directory (gitignored) so the written results can be inspected.
OUTPUT_DIR = Path(__file__).parent / "select_rep_output"


class TestCircularMean:
    """Tests for :func:`circular_mean`."""

    def test_wikipedia_example(self):
        """355, 5, 15 degrees should have a circular mean of ~5 degrees."""
        angles = np.deg2rad([355.0, 5.0, 15.0])
        mean_deg = math.degrees(circular_mean(angles))
        assert abs(mean_deg - 5.0) < 1e-6

    def test_wraparound_zero(self):
        """350 and 10 degrees should average to ~0 (not 180)."""
        angles = np.deg2rad([350.0, 10.0])
        mean_deg = math.degrees(circular_mean(angles))
        assert abs(((mean_deg + 180) % 360) - 180) < 1e-6

    def test_identical_angles(self):
        """Circular mean of identical angles is that angle."""
        angles = np.deg2rad([42.0, 42.0, 42.0])
        assert abs(math.degrees(circular_mean(angles)) - 42.0) < 1e-6

    def test_axis_reduction(self):
        """Mean is computed along the requested axis."""
        angles = np.deg2rad(np.array([[0.0, 90.0], [0.0, 90.0]]))
        means = np.rad2deg(circular_mean(angles, axis=0))
        assert means.shape == (2,)
        assert abs(means[0] - 0.0) < 1e-6
        assert abs(means[1] - 90.0) < 1e-6


class TestComputeFrameDeviations:
    """Tests for :func:`compute_frame_deviations`."""

    def test_zero_when_equal_to_mean(self):
        """A frame equal to the means has zero deviation."""
        means = np.deg2rad([10.0, -120.0, 75.0])
        torsions = means[np.newaxis, :].copy()
        dev = compute_frame_deviations(torsions, means)
        assert dev.shape == (1,)
        assert abs(dev[0]) < 1e-12

    def test_symmetry(self):
        """Deviation is symmetric about the mean (+/- delta give same value)."""
        means = np.array([0.0])
        delta = 0.5
        torsions = np.array([[delta], [-delta]])
        dev = compute_frame_deviations(torsions, means)
        assert abs(dev[0] - dev[1]) < 1e-12
        assert abs(dev[0] - (1.0 - math.cos(delta))) < 1e-12

    def test_range(self):
        """Deviation lies within [0, 2]."""
        rng = np.random.default_rng(0)
        torsions = rng.uniform(-math.pi, math.pi, size=(50, 4))
        means = circular_mean(torsions, axis=0)
        dev = compute_frame_deviations(torsions, means)
        assert np.all(dev >= 0.0)
        assert np.all(dev <= 2.0)


class TestFindRotatableTorsions:
    """Tests for :func:`find_rotatable_torsions`."""

    def test_returns_valid_torsions(self):
        mol = Chem.SDMolSupplier(str(LIGAND_SDF), removeHs=False)[0]
        assert mol is not None
        n_atoms = mol.GetNumAtoms()

        torsions = find_rotatable_torsions(mol)
        assert len(torsions) > 0

        for t in torsions:
            assert len(t) == 4
            assert len(set(t)) == 4
            for idx in t:
                assert 0 <= idx < n_atoms
            # No hydrogen terminals.
            assert mol.GetAtomWithIdx(t[0]).GetAtomicNum() > 1
            assert mol.GetAtomWithIdx(t[3]).GetAtomicNum() > 1

    def test_unique_central_bonds(self):
        """At most one torsion per central bond."""
        mol = Chem.SDMolSupplier(str(LIGAND_SDF), removeHs=False)[0]
        torsions = find_rotatable_torsions(mol)
        central_bonds = [(min(b, c), max(b, c)) for _, b, c, _ in torsions]
        assert len(central_bonds) == len(set(central_bonds))


class TestSelectRepresentativeFrame:
    """Integration tests for :func:`select_representative_frame`."""

    def test_returns_minimum_deviation_frame(self):
        result = select_representative_frame(
            topology=TOPOLOGY,
            trajectory=TRAJECTORY,
            ligand_sdf=LIGAND_SDF,
        )
        deviations = result["deviations"]
        rep_frame = result["rep_frame"]
        n_frames = deviations.shape[0]

        assert 0 <= rep_frame < n_frames
        assert rep_frame == int(np.argmin(deviations))
        assert deviations[rep_frame] == pytest.approx(np.min(deviations))

        torsions = result["torsions"]
        means = result["means"]
        assert torsions.shape[0] == n_frames
        assert torsions.shape[1] == len(result["torsion_atoms"])
        assert means.shape[0] == len(result["torsion_atoms"])
        assert np.all(torsions >= -math.pi - 1e-6)
        assert np.all(torsions <= math.pi + 1e-6)

    def test_writes_outputs(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_pdb = OUTPUT_DIR / "rep_protein.pdb"
        out_sdf = OUTPUT_DIR / "rep_ligand.sdf"
        out_fig = OUTPUT_DIR / "torsions.png"

        result = select_representative_frame(
            topology=TOPOLOGY,
            trajectory=TRAJECTORY,
            ligand_sdf=LIGAND_SDF,
            out_pdb=out_pdb,
            out_sdf=out_sdf,
            out_fig=out_fig,
        )

        assert out_pdb.is_file()
        assert out_sdf.is_file()
        assert out_fig.is_file()

        ref_mol = Chem.SDMolSupplier(str(LIGAND_SDF), removeHs=False)[0]
        out_mol = Chem.SDMolSupplier(str(out_sdf), removeHs=False)[0]
        assert out_mol is not None
        assert out_mol.GetNumAtoms() == ref_mol.GetNumAtoms()

        # Written ligand coordinates must match the representative-frame coordinates.
        rep_frame = result["rep_frame"]
        import MDAnalysis as mda

        u = mda.Universe(str(TOPOLOGY), str(TRAJECTORY))
        u.trajectory[rep_frame]
        lig_pos = u.atoms[: ref_mol.GetNumAtoms()].positions
        out_pos = out_mol.GetConformer().GetPositions()
        assert np.allclose(out_pos, lig_pos, atol=1e-2)

    def test_ligand_selection_mismatch_raises(self):
        with pytest.raises(ValueError, match="atom ordering"):
            select_representative_frame(
                topology=TOPOLOGY,
                trajectory=TRAJECTORY,
                ligand_sdf=LIGAND_SDF,
                ligand_selection="protein",
            )
