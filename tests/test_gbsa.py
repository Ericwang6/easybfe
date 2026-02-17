import os, shutil
from pathlib import Path

import numpy as np
import pytest

from easybfe.core.protein import Protein
from easybfe.core.ligand import Ligand, LigandLoader
from easybfe.gbsa import GBSARunner
from easybfe.gbsa.amber import run_gbsa_for_ligand_conformers
from easybfe.smff import load_parametrizer


def _load_test_protein_and_ligand():
    """Helper to load tyk2 protein and jmc_23 ligand from test data."""
    base_dir = Path(__file__).parent
    protein_pdb = base_dir / "data" / "tyk2_amber_h.pdb"
    ligand_sdf = base_dir / "data" / "jmc_23.sdf"

    assert protein_pdb.exists(), f"Missing test protein file: {protein_pdb}"
    assert ligand_sdf.exists(), f"Missing test ligand file: {ligand_sdf}"

    protein = Protein.from_pdb(protein_pdb, name="tyk2")
    
    # Load ligand
    loader = LigandLoader()
    ligands = loader.load(ligand_sdf, only_first=True, use_stem_as_name=True)
    assert len(ligands) == 1
    ligand = ligands[0]
    
    # Parametrize ligand (required for GBSARunner)
    smff = load_parametrizer('gaff2', 'gas')
    ligand = smff.run(ligand)
    
    # Verify ligand has required auxiliary files
    assert 'xml' in ligand.auxiliary_files
    assert 'pdb' in ligand.auxiliary_files
    
    return protein, ligand


def test_gbsa_runner_initialization():
    """Test GBSARunner can be initialized with protein and ligand."""
    protein, ligand = _load_test_protein_and_ligand()
    
    runner = GBSARunner(protein, ligand, igb=2)
    
    # Check that systems and contexts are created
    assert runner.ligand_system is not None
    assert runner.ligand_ctx is not None
    assert runner.protein_system is not None
    assert runner.protein_ctx is not None
    assert runner.complex_system is not None
    assert runner.complex_ctx is not None


def test_gbsa_runner_igb_values():
    """Test GBSARunner with different igb values."""
    protein, ligand = _load_test_protein_and_ligand()
    
    # Test supported igb values
    for igb in [1, 2, 5, 7, 8]:
        runner = GBSARunner(protein, ligand, igb=igb)
        assert runner is not None
    
    # Test unsupported igb value
    with pytest.raises(ValueError, match="Unsupported igb value"):
        GBSARunner(protein, ligand, igb=3)


def test_gbsa_runner_missing_xml():
    """Test GBSARunner raises error when ligand XML is missing."""
    protein, ligand = _load_test_protein_and_ligand()
    
    # Remove XML from ligand
    ligand.auxiliary_files.pop('xml', None)
    
    with pytest.raises(ValueError, match="ligand.auxiliary_files\['xml'\] not found"):
        GBSARunner(protein, ligand)


def test_gbsa_compute_single_frame():
    """Test compute_single_frame returns a binding energy."""
    protein, ligand = _load_test_protein_and_ligand()
    
    runner = GBSARunner(protein, ligand, igb=2)
    
    # Extract positions
    protein_pdb = protein.to_openmm()
    protein_pos = np.array([[v.x, v.y, v.z] for v in protein_pdb.positions]) * 10  # Convert to Angstroms
    
    ligand_pdb = ligand.to_openmm()
    ligand_pos = np.array([[v.x, v.y, v.z] for v in ligand_pdb.positions]) * 10  # Convert to Angstroms
    
    # Compute binding energy
    binding_energy = runner.compute_single_frame(protein_pos, ligand_pos)
    print(f"Binding energy: {binding_energy} kJ/mol ({binding_energy/4.184} kcal/mol)")
    
    # Check that it returns a float
    assert isinstance(binding_energy, float)
    
    # Binding energy should be reasonable (typically negative for favorable binding)
    # But we don't enforce a specific range as it depends on the system
    assert not np.isnan(binding_energy)
    assert not np.isinf(binding_energy)


def test_gbsa_compute_multiple_frames():
    """Test compute_multiple_frames returns array of binding energies."""
    protein, ligand = _load_test_protein_and_ligand()
    
    runner = GBSARunner(protein, ligand, igb=2)
    
    # Extract positions
    protein_pdb = protein.to_openmm()
    protein_pos = np.array([[v.x, v.y, v.z] for v in protein_pdb.positions]) * 10
    
    ligand_pdb = ligand.to_openmm()
    ligand_pos = np.array([[v.x, v.y, v.z] for v in ligand_pdb.positions]) * 10
    
    # Create multiple frames (just duplicate the same frame for testing)
    n_frames = 3
    protein_positions = np.tile(protein_pos[None, :, :], (n_frames, 1, 1))
    ligand_positions = np.tile(ligand_pos[None, :, :], (n_frames, 1, 1))
    
    # Compute binding energies with progress bar
    energies = runner.compute_multiple_frames(protein_positions, ligand_positions, progress_bar=True)
    
    # Check output
    assert isinstance(energies, np.ndarray)
    assert energies.shape == (n_frames,)
    assert energies.dtype == np.float64
    
    # All energies should be finite
    assert np.all(np.isfinite(energies))
    
    # Since we're using the same frame, energies should be identical
    assert np.allclose(energies, energies[0], rtol=1e-6)


def test_gbsa_compute_multiple_frames_no_progress_bar():
    """Test compute_multiple_frames with progress_bar=False."""
    protein, ligand = _load_test_protein_and_ligand()
    
    runner = GBSARunner(protein, ligand, igb=2)
    
    # Extract positions
    protein_pdb = protein.to_openmm()
    protein_pos = np.array([[v.x, v.y, v.z] for v in protein_pdb.positions]) * 10
    
    ligand_pdb = ligand.to_openmm()
    ligand_pos = np.array([[v.x, v.y, v.z] for v in ligand_pdb.positions]) * 10
    
    # Create multiple frames
    n_frames = 2
    protein_positions = np.tile(protein_pos[None, :, :], (n_frames, 1, 1))
    ligand_positions = np.tile(ligand_pos[None, :, :], (n_frames, 1, 1))
    
    # Compute binding energies without progress bar
    energies = runner.compute_multiple_frames(protein_positions, ligand_positions, progress_bar=False)
    
    # Check output
    assert isinstance(energies, np.ndarray)
    assert energies.shape == (n_frames,)
    assert np.all(np.isfinite(energies))


def test_gbsa_compute_multiple_frames_shape_validation():
    """Test compute_multiple_frames validates input shapes."""
    protein, ligand = _load_test_protein_and_ligand()
    
    runner = GBSARunner(protein, ligand, igb=2)
    
    # Extract positions
    protein_pdb = protein.to_openmm()
    protein_pos = np.array([[v.x, v.y, v.z] for v in protein_pdb.positions]) * 10
    
    ligand_pdb = ligand.to_openmm()
    ligand_pos = np.array([[v.x, v.y, v.z] for v in ligand_pdb.positions]) * 10
    
    # Test wrong shape for protein_positions
    with pytest.raises(ValueError, match="protein_positions must have shape"):
        runner.compute_multiple_frames(protein_pos, ligand_pos[None, :, :])
    
    # Test wrong shape for ligand_positions
    with pytest.raises(ValueError, match="ligand_positions must have shape"):
        runner.compute_multiple_frames(protein_pos[None, :, :], ligand_pos)
    
    # Test mismatched number of frames
    protein_positions = np.tile(protein_pos[None, :, :], (2, 1, 1))
    ligand_positions = np.tile(ligand_pos[None, :, :], (3, 1, 1))
    with pytest.raises(ValueError, match="Number of frames must match"):
        runner.compute_multiple_frames(protein_positions, ligand_positions)


def test_gbsa_amber():
    base_dir = Path(__file__).parent
    protein_pdb = base_dir / "data" / "tyk2_amber.pdb"
    ligand_sdf = base_dir / "data" / "jmc_23.sdf"
    gbsa_dir = base_dir / '_test_gbsa_amber'
    if gbsa_dir.is_dir():
        shutil.rmtree(gbsa_dir)
    run_gbsa_for_ligand_conformers(
        protein_pdb, ligand_sdf, ligand_confs=[ligand_sdf], wdir = gbsa_dir, charge_method='gas', run_em=False
    )
