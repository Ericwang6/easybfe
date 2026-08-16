"""A .ligpack archive is interchangeable with a ligand directory across the CLI.

Covers ``easybfe ligand pargen`` (ligpack as an output format) and the three
consumers of a parameterized ligand: ``abfe setup``, ``abfe pipeline`` and
``rbfe setup``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from easybfe.cli.main import main
from easybfe.core.ligand import Ligand


DATA = Path(__file__).parent / "data"
AUX = {"prmtop", "inpcrd", "pdb", "xml"}


def _parametrize(sdf: Path) -> Ligand:
    """Parameterize with gaff2/gas: cheap charges, complete auxiliary files."""
    from easybfe.smff import load_parametrizer

    return load_parametrizer("gaff2", "gas").run(Ligand.from_file(sdf, only_first=True))


@pytest.fixture(scope="module")
def jmc_23() -> Ligand:
    return _parametrize(DATA / "jmc_23.sdf")


@pytest.fixture(scope="module")
def jmc_32() -> Ligand:
    return _parametrize(DATA / "jmc_32.sdf")


def _invoke(*args: str):
    result = CliRunner().invoke(main, list(args))
    assert result.exit_code == 0, result.output or str(result.exception)
    return result


def test_pargen_writes_a_single_ligpack(tmp_path: Path):
    """``-o out.ligpack`` writes an archive instead of a directory."""
    out = tmp_path / "benzene.ligpack"
    _invoke(
        "ligand", "pargen", str(DATA / "benzene.sdf"),
        "-o", str(out), "-f", "gaff2", "-c", "gas", "-n", "1",
    )

    assert out.is_file()
    ligand = Ligand.from_ligpack(out)
    assert ligand.name == "benzene"
    assert AUX <= set(ligand.auxiliary_files)
    # The scratch directory has no output directory to live in, but it is still
    # cleaned up rather than left beside the archive.
    assert list(tmp_path.iterdir()) == [out]


def test_pargen_writes_one_ligpack_per_ligand(tmp_path: Path):
    """``--ligpack`` with ``-O`` replaces per-ligand subdirectories with archives."""
    out_base = tmp_path / "ligands"
    _invoke(
        "ligand", "pargen", str(DATA / "benzene.sdf"), str(DATA / "chlorobenzene.sdf"),
        "-O", str(out_base), "--ligpack", "-f", "gaff2", "-c", "gas", "-n", "1",
    )

    assert sorted(p.name for p in out_base.iterdir()) == [
        "benzene.ligpack", "chlorobenzene.ligpack",
    ]
    assert Ligand.from_ligpack(out_base / "chlorobenzene.ligpack").name == "chlorobenzene"


def test_abfe_setup_accepts_a_ligpack(tmp_path: Path, jmc_23: Ligand):
    """``abfe setup --ligand x.ligpack`` builds the same three legs as a directory."""
    ligpack = jmc_23.dump_ligpack(tmp_path / "jmc_23.ligpack")
    output_dir = tmp_path / "abfe_output"

    _invoke(
        "abfe", "setup", str(DATA / "config_abfe.json"),
        "--ligand", str(ligpack),
        "--protein", str(DATA / "tyk2_pdbfixer.pdb"),
        "--output", str(output_dir),
    )

    for leg in ("solvent", "complex", "restraint"):
        assert (output_dir / leg / "system.prmtop").is_file()
        assert (output_dir / leg / "system.inpcrd").is_file()
        assert (output_dir / leg / "run.sh").is_file()


def test_abfe_setup_output_dir_is_named_without_the_suffix(tmp_path: Path, monkeypatch, jmc_23: Ligand):
    """Under ``output_base``, ``jmc_23.ligpack`` runs in ``jmc_23/``, not ``jmc_23.ligpack/``."""
    from easybfe.abfe.config import AmberAbfeConfig
    from easybfe.amber import prep_ligand_abfe

    ligpack = jmc_23.dump_ligpack(tmp_path / "jmc_23.ligpack")
    captured: dict = {}
    monkeypatch.setattr(prep_ligand_abfe, "setup_ligand_abfe", lambda **kwargs: captured.update(kwargs))

    cfg = AmberAbfeConfig.model_validate(
        {
            "protein": DATA / "tyk2_pdbfixer.pdb",
            "ligand": str(ligpack),
            "output_base": tmp_path / "runs",
        }
    )
    prep_ligand_abfe.setup_ligand_abfe_from_config(cfg)

    assert captured["output_dir"] == (tmp_path / "runs" / "jmc_23").resolve()
    assert captured["ligand"].name == "jmc_23"
    assert AUX <= set(captured["ligand"].auxiliary_files)


def test_abfe_pipeline_loads_a_ligpack_instead_of_reparameterizing(tmp_path: Path, jmc_23: Ligand):
    """The pipeline treats a ligpack as an already-parameterized input."""
    from easybfe.abfe.piepline import ABFE

    ligpack = jmc_23.dump_ligpack(tmp_path / "jmc_23.ligpack")
    runner = ABFE(
        DATA / "config_abfe.json",
        protein=DATA / "tyk2_pdbfixer.pdb",
        ligand=ligpack,
        output=tmp_path / "run",
    )
    try:
        ligand = runner.prepare_ligand()
    finally:
        runner.close()

    assert ligand.name == "jmc_23"
    # Identical aux files prove they were read from the archive, not regenerated.
    assert ligand.auxiliary_files == jmc_23.auxiliary_files
    assert (tmp_path / "run" / "ligand" / "jmc_23.prmtop").is_file()


def test_rbfe_setup_accepts_ligpacks(tmp_path: Path, monkeypatch, jmc_23: Ligand, jmc_32: Ligand):
    """Both ends of an RBFE edge can be ligpacks; the edge directory drops the suffix."""
    from easybfe.amber import prep_ligand_rbfe
    from easybfe.config.amber.rbfe import AmberLigandRbfeConfig

    pack_a = jmc_23.dump_ligpack(tmp_path / "jmc_23.ligpack")
    pack_b = jmc_32.dump_ligpack(tmp_path / "jmc_32.ligpack")
    captured: dict = {}
    monkeypatch.setattr(prep_ligand_rbfe, "setup_ligand_rbfe", lambda **kwargs: captured.update(kwargs))

    cfg = AmberLigandRbfeConfig.model_validate(
        {
            "protein": DATA / "tyk2_pdbfixer.pdb",
            "ligandA": str(pack_a),
            "ligandB": str(pack_b),
            "output_base": tmp_path / "rbfe",
        }
    )
    prep_ligand_rbfe.setup_ligand_rbfe_from_config(cfg)

    assert captured["ligandA"].name == "jmc_23"
    assert captured["ligandB"].name == "jmc_32"
    assert AUX <= set(captured["ligandA"].auxiliary_files)
    assert AUX <= set(captured["ligandB"].auxiliary_files)
    assert captured["output_dir"] == (tmp_path / "rbfe" / "jmc_23~jmc_32").resolve()
