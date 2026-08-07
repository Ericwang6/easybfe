"""Tests for :mod:`easybfe.amber.workflow` command rendering."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from easybfe.config.amber.simulation import default_fep_workflow
from easybfe.amber.workflow import Step, Workflow, create_groupfile_from_steps


def _leg(tmp_path: Path, n_lambdas: int = 2) -> list[Workflow]:
    """A leg directory with ``n_lambdas`` per-lambda workflows created on disk."""
    (tmp_path / "system.prmtop").touch()
    (tmp_path / "system.inpcrd").touch()
    workflows = []
    for n in range(n_lambdas):
        wf = Workflow(
            wdir=tmp_path / f"lambda{n}",
            prmtop=tmp_path / "system.prmtop",
            inpcrd=tmp_path / "system.inpcrd",
            steps=[Step(config=cfg) for cfg in default_fep_workflow()],
        )
        wf.create()
        workflows.append(wf)
    return workflows


def test_step_shell_is_multiline_and_chains_ambpdb(tmp_path: Path) -> None:
    """The per-stage script is meant to be read and edited by hand."""
    workflows = _leg(tmp_path, n_lambdas=1)
    first = next(iter(workflows[0].steps))
    text = (workflows[0].steps[first].wdir / f"{first}.sh").read_text()

    assert text.count("\n") > 5, "arguments should be one per line, not one long line"
    # `&&`, not a newline: a failed pmemd must not be followed by ambpdb.
    assert "&& ambpdb" in text


def test_devnull_stays_absolute(tmp_path: Path) -> None:
    """``-l`` points at the device, not at a ladder of ``../``."""
    workflows = _leg(tmp_path)
    last = list(workflows[0].steps)[-1]
    step = workflows[0].steps[last]

    shell = step.render_shell()
    groupfile_line = step.render_groupfile_line(relative_to=tmp_path)

    assert f"-l {os.devnull}" in shell
    assert f"-l {os.devnull}" in groupfile_line
    assert "../dev/null" not in shell
    assert "../dev/null" not in groupfile_line


def test_groupfile_has_one_line_per_step(tmp_path: Path) -> None:
    """AMBER reads a groupfile as one group per line — this must not wrap."""
    workflows = _leg(tmp_path, n_lambdas=3)
    last = list(workflows[0].steps)[-1]
    steps = [wf.steps[last] for wf in workflows]

    text = create_groupfile_from_steps(steps, tmp_path)

    lines = text.strip().split("\n")
    assert len(lines) == 3
    for line, wf in zip(lines, workflows):
        assert line.startswith("-O ")
        # The executable name belongs on the mpirun command line, not here.
        assert "pmemd" not in line
        assert f"lambda{workflows.index(wf)}/" in line


def test_step_renderers_agree_on_arguments(tmp_path: Path) -> None:
    """Both renderers come from one argument list, so they cannot drift."""
    workflows = _leg(tmp_path)
    last = list(workflows[0].steps)[-1]
    step = workflows[0].steps[last]

    args = step.create_args(relative_to=tmp_path)
    groupfile_line = step.render_groupfile_line(relative_to=tmp_path)

    assert groupfile_line == " ".join(args)


def test_workflow_creates_per_lambda_debug_script(tmp_path: Path) -> None:
    """``lambda*/run.sh`` is kept: it is how a single window is re-run by hand."""
    workflows = _leg(tmp_path, n_lambdas=1)
    run_sh = workflows[0].wdir / "run.sh"

    assert run_sh.is_file()
    assert os.access(run_sh, os.X_OK)
    for name in workflows[0].steps:
        assert f"source {name}.sh" in run_sh.read_text()


def test_workflow_no_longer_writes_run_submit(tmp_path: Path) -> None:
    """The old ``run.submit`` was a stub with an always-empty header."""
    workflows = _leg(tmp_path, n_lambdas=1)
    assert not (workflows[0].wdir / "run.submit").exists()
