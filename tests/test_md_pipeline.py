"""``easybfe md pipeline`` orchestration, without running MD.

The real MD is exercised on a GPU node (examples/md-pipeline/md.slurm); here the
``run.sh`` execution and the trajectory analysis are stubbed out.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from easybfe.config.amber.simulation import AmberPlainMDConfig
from easybfe.md import MD, MDPipelineConfig


DATA = Path(__file__).parent / "data"
PROTEIN = DATA / "tyk2_pdbfixer_dry.pdb"
LIGAND = DATA / "jmc_23.sdf"


def _config() -> MDPipelineConfig:
    # A short workflow keeps setup cheap; nothing here runs it.
    return MDPipelineConfig.model_validate({
        "simulation": {
            "buffer": 10.0,
            "workflow": [
                {"type": "em", "name": "01.em", "use_remd": False, "use_mpi": False},
                {"type": "prod", "name": "02.prod", "use_remd": False, "use_mpi": False},
            ],
        },
    })


def _stub_md(monkeypatch, runner: MD, calls: list) -> None:
    """Replace the MD run and analysis with writers of the files they produce."""
    prod = runner.root / runner.prod_name

    def fake_run_script(directory, *args, **kwargs):
        calls.append(("run", Path(directory)))
        prod.mkdir(exist_ok=True)
        (prod / f"{runner.prod_name}.mdcrd").write_text("")
        return True

    def fake_analysis(directory, **kwargs):
        calls.append(("analyze", Path(directory)))
        np.savetxt(prod / "prod_rmsd.txt", [[0.01, 0.5], [0.02, 1.5]], header="Time (ns), RMSD (angstrom)")
        if (Path(directory) / "protein.pdb").is_file() and (Path(directory) / "ligand").is_dir():
            np.savetxt(prod / "gbsa.txt", [-41.84, -83.68])

    monkeypatch.setattr("easybfe.md.pipeline.run_script", fake_run_script)
    monkeypatch.setattr("easybfe.analysis.plain_md.run_plain_md_analysis_workflow", fake_analysis)


@pytest.mark.parametrize(
    "protein, ligand, task_type",
    [(PROTEIN, LIGAND, "complex"), (PROTEIN, None, "protein"), (None, LIGAND, "ligand")],
)
def test_task_type_follows_inputs(tmp_path, protein, ligand, task_type) -> None:
    runner = MD(_config(), protein=protein, ligand=ligand, output=tmp_path / "run")
    try:
        assert runner.task_type == task_type
    finally:
        runner.close()


def test_needs_protein_or_ligand(tmp_path) -> None:
    with pytest.raises(ValueError, match="protein and ligand"):
        MD(_config(), output=tmp_path / "run")


def test_protein_only_pipeline(tmp_path, monkeypatch) -> None:
    runner = MD(_config(), protein=PROTEIN, output=tmp_path / "run")
    calls: list = []
    _stub_md(monkeypatch, runner, calls)

    result = runner.run()

    root = tmp_path / "run"
    assert [c[0] for c in calls] == ["run", "analyze"]
    assert (root / "system.prmtop").is_file() and (root / "run.sh").is_file()
    assert not (root / "ligand").exists()

    # config.json is an ordinary `md setup` config, with protein-only analysis defaults.
    md_config = AmberPlainMDConfig.model_validate(json.loads((root / "config.json").read_text()))
    assert md_config.task_type == "protein"
    assert md_config.analysis.rmsd_selection == "backbone"
    assert not md_config.analysis.interaction_analysis
    assert not md_config.analysis.do_gbsa

    assert result["task_type"] == "protein"
    assert result["rmsd"]["mean"] == pytest.approx(1.0)
    assert result["rmsd"]["final"] == pytest.approx(1.5)
    assert "gbsa" not in result
    assert json.loads((root / "result.json").read_text()) == result
    assert (root / "md.log").is_file()


def test_rerun_resumes_without_rebuilding(tmp_path, monkeypatch) -> None:
    root = tmp_path / "run"
    runner = MD(_config(), protein=PROTEIN, output=root)
    _stub_md(monkeypatch, runner, [])
    runner.run()
    prmtop_mtime = (root / "system.prmtop").stat().st_mtime_ns

    runner = MD(_config(), protein=PROTEIN, output=root)
    calls: list = []
    _stub_md(monkeypatch, runner, calls)
    monkeypatch.setattr(MD, "setup", lambda self: pytest.fail("setup must not re-run"))
    runner.run()

    assert [c[0] for c in calls] == ["run", "analyze"]
    assert (root / "system.prmtop").stat().st_mtime_ns == prmtop_mtime


def test_failed_md_raises(tmp_path, monkeypatch) -> None:
    runner = MD(_config(), protein=PROTEIN, output=tmp_path / "run")
    monkeypatch.setattr("easybfe.md.pipeline.run_script", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="MD failed"):
        runner.run()


def test_explicit_analysis_settings_win_over_type_defaults(tmp_path) -> None:
    config = _config()
    config.analysis = type(config.analysis).model_validate({"do_gbsa": True})
    runner = MD(config, protein=PROTEIN, output=tmp_path / "run")
    try:
        runner.setup()
    finally:
        runner.close()
    analysis = json.loads((tmp_path / "run" / "config.json").read_text())["analysis"]
    assert analysis["do_gbsa"] is True
    assert analysis["interaction_analysis"] is False


@pytest.mark.parametrize("box_shape", ["cube", "dodecahedron", "octahedron"])
def test_small_solute_box_fits_pmemd_pairlist(tmp_path, box_shape) -> None:
    """A lone ligand with a modest buffer used to get a box pmemd rejects
    ("max pairlist cutoff must be less than unit cell max sphere radius")."""
    import parmed
    from easybfe.amber.prep_utils import inscribedSphereRadius

    config = _config()
    config.simulation.buffer = 15.0
    config.simulation.box_shape = box_shape
    runner = MD(config, ligand=LIGAND, output=tmp_path / "run")
    try:
        runner.prepare_ligand()
        runner.setup()
    finally:
        runner.close()

    root = tmp_path / "run"
    box = parmed.load_file(str(root / "system.prmtop"), xyz=str(root / "system.inpcrd")).box_vectors
    box = box.value_in_unit(parmed.unit.angstrom)
    pairlist_cut = max(step.cntrl.cut for step in config.simulation.workflow) + 2.0  # + skinnb
    # pmemd.cuda: >= 3 neighbor-list cells of width cut + skinnb across the box
    # ("Small box detected, with <= 2 cells").
    assert 2 * inscribedSphereRadius(box) >= 3 * pairlist_cut
