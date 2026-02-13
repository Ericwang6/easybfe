"""Tests for the ABFE CLI (easybfe.cli.abfe): setup and analyze commands.

"""
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from easybfe.cli.main import main


def test_abfe_setup_cli():
    """Run ABFE setup via CLI (mimics test_ligand_abfe flow with config + overrides)."""
    test_dir = Path(__file__).parent / "_test_cli_abfe"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    data_dir = Path(__file__).parent / "data"
    ligand_sdf = data_dir / "jmc_23.sdf"
    protein_pdb = data_dir / "tyk2_pdbfixer.pdb"

    from easybfe.core.ligand import LigandLoader
    from easybfe.smff import load_parametrizer

    loader = LigandLoader()
    ligands = loader.load(ligand_sdf, only_first=True, use_stem_as_name=True)
    assert len(ligands) == 1
    ligand = ligands[0]
    smff = load_parametrizer("gaff2", "gas")
    ligand = smff.run(ligand)
    assert "xml" in ligand.auxiliary_files
    assert "pdb" in ligand.auxiliary_files

    ligand_dir = test_dir / "jmc_23"
    ligand.dump(ligand_dir)

    config_path = data_dir / "config_abfe.json"
    output_dir = test_dir / "abfe_output"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "abfe",
            "setup",
            str(config_path),
            "--ligand",
            str(ligand_dir),
            "--protein",
            str(protein_pdb),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, (result.output or str(result.exception))
    assert (output_dir / "solvent").exists()
    assert (output_dir / "complex").exists()
    assert (output_dir / "restraint").exists()
    for leg_name in ["solvent", "complex", "restraint"]:
        leg_dir = output_dir / leg_name
        assert (leg_dir / "system.prmtop").exists()
        assert (leg_dir / "system.inpcrd").exists()
        assert (leg_dir / "lambda0").exists()
        assert (leg_dir / "run.sh").exists()


def test_abfe_setup_cli_usage_error():
    """Passing both --ligand and --ligand-batch raises UsageError."""
    test_dir = Path(__file__).parent / "_test_cli_abfe"
    data_dir = Path(__file__).parent / "data"
    config_path = data_dir / "config_abfe.json"
    test_dir.mkdir(exist_ok=True)
    ligand_dir = test_dir / "jmc_23"
    ligand_dir.mkdir(exist_ok=True)  # must exist so Click validates before our check
    batch_file = test_dir / "batch.txt"
    batch_file.write_text(str(ligand_dir) + "\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "abfe",
            "setup",
            str(config_path),
            "--ligand",
            str(ligand_dir),
            "--ligand-batch",
            str(batch_file),
        ],
    )

    assert result.exit_code != 0
    assert "Cannot set both" in result.output


def test_abfe_analyze_cli():
    """Run ABFE analyze on a directory with cached result.json; should return 0."""
    test_dir = Path(__file__).parent / "_test_cli_abfe_analyze"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    res = {
        "complex": -5.0,
        "complex_std": 0.1,
        "solvent": 2.0,
        "solvent_std": 0.1,
        "restraint": 0.5,
        "restraint_std": 0.05,
        "boresch": 3.0,
        "total": -9.5,
        "total_std": 0.15,
    }
    (test_dir / "result.json").write_text(json.dumps(res, indent=2))

    runner = CliRunner()
    result = runner.invoke(main, ["abfe", "analyze", str(test_dir)])

    assert result.exit_code == 0, (result.output or str(result.exception))


def test_abfe_analyze_cli_with_options():
    """ABFE analyze accepts --prod-prefix, --temperature, --force without error when result exists."""
    test_dir = Path(__file__).parent / "_test_cli_abfe_analyze"
    test_dir.mkdir(exist_ok=True)
    res = {"total": -9.5, "total_std": 0.15, "complex": -5.0, "complex_std": 0.1,
           "solvent": 2.0, "solvent_std": 0.1, "restraint": 0.5, "restraint_std": 0.05, "boresch": 3.0}
    (test_dir / "result.json").write_text(json.dumps(res))

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["abfe", "analyze", str(test_dir), "--prod-prefix", "05.prod", "--temperature", "298.15"],
    )
    assert result.exit_code == 0
