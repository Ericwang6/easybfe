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
