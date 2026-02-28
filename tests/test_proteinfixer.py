"""
Tests for protein fixer method.
"""
import os

from pathlib import Path
import pytest

from easybfe.protein_prep import ProteinFixer



@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / 'data' / 'proteinfixer'


def test_proteinfixer_run_1apm(test_data_dir):
    """Test ProteinFixer.run method with 1APM.pdb."""
    input_pdb = test_data_dir / '1APM.pdb'
    output_pdb = os.path.join(test_data_dir, '1APM_fixed.pdb')
    
    fixer = ProteinFixer(str(input_pdb), wizard=True)
    fixer.run(
        skip_missing_terminal_residues=False,
        max_num_consecutive_missing_residues=None,
        keep_water=True,
        keep_ions=True,
        pH=7.4,
        cap_gaps=True,
        force_cap_terminals=False,
        out=output_pdb
    )
    
    # Verify output file was created
    assert os.path.exists(output_pdb), "Output PDB file was not created"


def test_proteinfixer_run_5rob(test_data_dir):
    """Test ProteinFixer.run method with 5ROB.pdb."""
    input_pdb = test_data_dir / '5ROB.pdb'
    output_pdb = os.path.join(test_data_dir, '5ROB_fixed.pdb')
    
    fixer = ProteinFixer(str(input_pdb))
    fixer.run(
        output_protein=output_pdb,
        skip_missing_terminal_residues=True,
        max_num_consecutive_missing_residues=None,
        keep_water=True,
        keep_ions=True,
        pH=7.4,
        cap_gaps=True,
        force_cap_terminals=False,
    )
    
    # Verify output file was created
    assert os.path.exists(output_pdb), "Output PDB file was not created"

