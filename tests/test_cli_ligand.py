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
    """`ligand cdock` writes the pose to the path given by --output (-o) for a single probe."""
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
    out_sdf = WORK_DIR / "pose_out.sdf"

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
            str(out_sdf),
            "--no-em",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "vina_score:" in result.output
    assert "pose_sdf:" in result.output
    assert str(out_sdf) in result.output

    assert out_sdf.is_file()
    out_mol = next(m for m in Chem.SDMolSupplier(str(out_sdf), removeHs=False) if m is not None)
    assert out_mol.GetNumConformers() > 0


def test_ligand_cdock_cli_output_dir_multi_probe() -> None:
    """Multiple probes require --output-dir; each pose is DIR/<name>.sdf and .tmp holds Vina files."""
    if WORK_DIR.is_dir():
        shutil.rmtree(WORK_DIR)
    out_dir = WORK_DIR / "cdock_multi"
    out_dir.mkdir(parents=True)

    suppl = Chem.SDMolSupplier(str(DATA_DIR / "tyk2_ligands.sdf"), removeHs=False)
    mols = [m for m in suppl if m is not None]
    assert len(mols) >= 2
    probe_path = out_dir / "probes.sdf"
    with Chem.SDWriter(str(probe_path)) as writer:
        writer.write(deepcopy(mols[0]))
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
            "-O",
            str(out_dir),
            "--no-em",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert (out_dir / ".tmp").is_dir()
    written = list(out_dir.glob("*.sdf"))
    probe_only = [p for p in written if p.name != "probes.sdf"]
    assert len(probe_only) == 2
    for p in probe_only:
        mol = next(m for m in Chem.SDMolSupplier(str(p), removeHs=False) if m is not None)
        assert mol.GetNumConformers() > 0


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
            "-o",
            str(WORK_DIR / "pose_box.sdf"),
            "--box-center",
            "0.0",
            "0.0",
            "0.0",
        ],
    )
    assert result.exit_code != 0
