from pathlib import Path

from click.testing import CliRunner

from easybfe.cli.main import main


def test_protein_check_cli_default_forcefield():
    """`easybfe protein check` succeeds with default force fields."""
    data_dir = Path(__file__).parent / "data"
    protein_pdb = data_dir / "tyk2_pdbfixer.pdb"

    runner = CliRunner()
    result = runner.invoke(main, ["protein", "check", str(protein_pdb)])

    assert result.exit_code == 0, (result.output or str(result.exception))
    assert "Force field check passed." in result.output


def test_protein_summary_cli():
    """`easybfe protein summary` prints structure and quality summary."""
    data_dir = Path(__file__).parent / "data"
    protein_pdb = data_dir / "tyk2_pdbfixer.pdb"

    runner = CliRunner()
    result = runner.invoke(main, ["protein", "summary", str(protein_pdb)])

    assert result.exit_code == 0, (result.output or str(result.exception))
    assert "Structure summary for:" in result.output
    assert "Components" in result.output
    assert "Non-standard residues" in result.output
    assert "Missing residues" in result.output
    assert "Missing atoms" in result.output
