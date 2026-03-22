"""CLI tests for easybfe ligand commands."""

import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from click.testing import CliRunner
from rdkit import Chem

pytest.importorskip("vina")

from easybfe.cli.main import main

DATA_DIR = Path(__file__).parent / "data"
WORK_DIR = Path(__file__).parent / "_test_cli_ligand"


def test_ligand_cdock_cli_writes_sdf() -> None:
    """`ligand cdock` runs and writes pose SDF under output/dock/constr_dock when -o is set."""
    if WORK_DIR.is_dir():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    suppl = Chem.SDMolSupplier(str(DATA_DIR / "tyk2_ligands.sdf"), removeHs=False)
    mols = [m for m in suppl if m is not None]
    assert len(mols) >= 2
    probe_path = WORK_DIR / "probe.sdf"
    with Chem.SDWriter(str(probe_path)) as writer:
        writer.write(deepcopy(mols[1]))

    protein = DATA_DIR / "tyk2_pdbfixer.pdb"
    ref_sdf = DATA_DIR / "tyk2_ligands.sdf"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ligand",
            "cdock",
            str(probe_path),
            "-r",
            str(ref_sdf),
            "-p",
            str(protein),
            "-o",
            str(WORK_DIR),
            "--no-em",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "vina_score:" in result.output
    assert "pose_sdf:" in result.output

    dock_dir = WORK_DIR / "dock" / "constr_dock"
    assert dock_dir.is_dir()
    sdfs = list(dock_dir.glob("*.sdf"))
    assert len(sdfs) == 1
    out_mol = next(m for m in Chem.SDMolSupplier(str(sdfs[0]), removeHs=False) if m is not None)
    assert out_mol.GetNumConformers() > 0


def test_ligand_cdock_cli_box_requires_both() -> None:
    """Passing only --box-center without --box-size is a usage error."""
    if WORK_DIR.is_dir():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    suppl = Chem.SDMolSupplier(str(DATA_DIR / "tyk2_ligands.sdf"), removeHs=False)
    mols = [m for m in suppl if m is not None]
    assert len(mols) >= 2
    probe_path = WORK_DIR / "probe.sdf"
    with Chem.SDWriter(str(probe_path)) as writer:
        writer.write(deepcopy(mols[1]))

    protein = DATA_DIR / "tyk2_pdbfixer.pdb"
    ref_sdf = DATA_DIR / "tyk2_ligands.sdf"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ligand",
            "cdock",
            str(probe_path),
            "-r",
            str(ref_sdf),
            "-p",
            str(protein),
            "--box-center",
            "0.0",
            "0.0",
            "0.0",
        ],
    )
    assert result.exit_code != 0
