import os
import shutil
from pathlib import Path
import pytest

from easybfe.core.ligand import Ligand, LigandLoader
from easybfe.core.protein import Protein
from easybfe.config import AmberSimulationConfig
from easybfe.smff import load_parametrizer
from easybfe.amber.prep_plain_md import setup_plain_md


def test_setup_plain_md():
    """Test setup_plain_md with ligand and protein."""
    # Setup test directory
    test_dir = Path(__file__).parent / '_test_plain_md'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Get test data paths
    data_dir = Path(__file__).parent / 'data'
    ligand_sdf = data_dir / 'jmc_23.sdf'
    protein_pdb = data_dir / 'tyk2_pdbfixer.pdb'
    
    # Load and parametrize ligand
    loader = LigandLoader()
    ligands = loader.load(ligand_sdf, only_first=True, use_stem_as_name=True)
    assert len(ligands) == 1
    ligand = ligands[0]
    
    # Parametrize ligand (this adds xml, pdb, prmtop, inpcrd to auxiliary_files)
    smff = load_parametrizer('gaff2', 'gas')
    ligand = smff.run(ligand)
    
    # Verify ligand has required auxiliary files
    assert 'xml' in ligand.auxiliary_files
    assert 'pdb' in ligand.auxiliary_files
    
    # Load protein
    protein = Protein.from_pdb(protein_pdb, name='tyk2')
    
    # Create config with default settings
    config = AmberSimulationConfig()
    
    # Setup plain MD
    wdir = test_dir / 'system'
    wf = setup_plain_md(
        ligand=ligand,
        protein=protein,
        config=config,
        wdir=wdir,
    )
    
    # Verify output files were created
    assert (wdir / 'system.inpcrd').exists()
    assert (wdir / 'system.prmtop').exists()
    assert (wdir / 'system.pdb').exists()
    assert (wdir / f'{ligand.name}.sdf').exists()
    assert (wdir / f'{ligand.name}.pdb').exists()
    assert (wdir / f'{ligand.name}.xml').exists()
    
    # Cleanup
    # shutil.rmtree(test_dir)


def test_setup_plain_md_ligand_only():
    """Test setup_plain_md with ligand only (no protein)."""
    # Setup test directory
    test_dir = Path(__file__).parent / '_test_plain_md_ligand_only'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Get test data paths
    data_dir = Path(__file__).parent / 'data'
    ligand_sdf = data_dir / 'jmc_23.sdf'
    
    # Load and parametrize ligand
    loader = LigandLoader()
    ligands = loader.load(ligand_sdf, only_first=True, use_stem_as_name=True)
    ligand = ligands[0]
    
    # Parametrize ligand
    smff = load_parametrizer('gaff2', 'gas')
    ligand = smff.run(ligand)
    
    # Create config
    config = AmberSimulationConfig()
    
    # Setup plain MD with ligand only
    wdir = test_dir / 'system'
    wf = setup_plain_md(
        ligand=ligand,
        protein=None,
        config=config,
        wdir=wdir,
    )
    
    # Verify output files were created
    assert (wdir / 'system.inpcrd').exists()
    assert (wdir / 'system.prmtop').exists()
    assert (wdir / 'system.pdb').exists()
    
    # Cleanup
    shutil.rmtree(test_dir)


def test_setup_plain_md_gas_phase():
    """Test setup_plain_md in gas phase (no solvent)."""
    # Setup test directory
    test_dir = Path(__file__).parent / '_test_plain_md_gas'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Get test data paths
    data_dir = Path(__file__).parent / 'data'
    ligand_sdf = data_dir / 'jmc_23.sdf'
    protein_pdb = data_dir / 'tyk2_pdbfixer.pdb'
    
    # Load and parametrize ligand
    loader = LigandLoader()
    ligands = loader.load(ligand_sdf, only_first=True, use_stem_as_name=True)
    ligand = ligands[0]
    
    # Parametrize ligand
    smff = load_parametrizer('gaff2', 'gas')
    ligand = smff.run(ligand)
    
    # Load protein
    protein = Protein.from_pdb(protein_pdb, name='tyk2')
    
    # Create config with gas_phase=True
    config = AmberSimulationConfig(gas_phase=True)
    
    # Setup plain MD
    wdir = test_dir / 'system'
    wf = setup_plain_md(
        ligand=ligand,
        protein=protein,
        config=config,
        wdir=wdir,
    )
    
    # Verify output files were created
    assert (wdir / 'system.inpcrd').exists()
    assert (wdir / 'system.prmtop').exists()
    assert (wdir / 'system.pdb').exists()
    
    # Cleanup
    shutil.rmtree(test_dir)


def test_setup_plain_md_error_both_none():
    """Test setup_plain_md raises error when both ligand and protein are None."""
    test_dir = Path(__file__).parent / '_test_plain_md_error'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    config = AmberSimulationConfig()
    wdir = test_dir / 'system'
    
    # Should raise RuntimeError when both are None
    with pytest.raises(RuntimeError, match="Both ligand and protein are None"):
        setup_plain_md(
            ligand=None,
            protein=None,
            config=config,
            wdir=wdir
        )
    
    # Cleanup
    shutil.rmtree(test_dir)
