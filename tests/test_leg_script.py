"""Tests for the generated per-leg ``run.sh``.

The script is executed inside a plain AMBER environment, so these tests pin the
two properties that make that possible: it must be valid bash, and it must not
reach for easybfe or python.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from easybfe.config.amber.simulation import default_fep_workflow
from easybfe.amber.workflow import Step, Workflow, create_script_for_workflows


STUB_MPIRUN = """#!/usr/bin/env bash
echo "mpirun $*"
if [ -n "$STUB_FAIL_STAGE" ] && printf '%s' "$*" | grep -q "$STUB_FAIL_STAGE"; then
  if [ "$STUB_BINARY_LOG" = "1" ]; then printf 'garbled\000\000output\n'; fi
  echo "cuStreamCreate return value: 709"
  exit 1
fi
exit 0
"""

STUB_PMEMD = """#!/usr/bin/env bash
echo "pmemd $*"
exit 0
"""


@pytest.fixture
def leg(tmp_path: Path) -> Path:
    """A leg directory with generated scripts, ready to run against stubs."""
    (tmp_path / "system.prmtop").touch()
    (tmp_path / "system.inpcrd").touch()
    workflows = []
    for n in range(2):
        workflows.append(
            Workflow(
                wdir=tmp_path / f"lambda{n}",
                prmtop=tmp_path / "system.prmtop",
                inpcrd=tmp_path / "system.inpcrd",
                steps=[Step(config=cfg) for cfg in default_fep_workflow()],
            )
        )
    create_script_for_workflows(workflows, tmp_path, -1)
    return tmp_path


@pytest.fixture
def stub_env(tmp_path_factory) -> dict:
    """Environment with stand-ins for mpirun/pmemd/ambpdb on PATH."""
    bindir = tmp_path_factory.mktemp("stubbin")
    for name, body in (
        ("mpirun", STUB_MPIRUN),
        ("pmemd.cuda", STUB_PMEMD),
        ("pmemd.cuda.MPI", STUB_PMEMD),
        ("ambpdb", STUB_PMEMD),
    ):
        path = bindir / name
        path.write_text(body)
        path.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    env["EASYBFE_DISABLE_MPS"] = "1"
    return env


def run_leg(leg: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "run.sh", *args],
        cwd=leg, env=env, capture_output=True, text=True,
    )


def test_generated_script_is_valid_bash(leg: Path) -> None:
    assert subprocess.run(["bash", "-n", "run.sh"], cwd=leg).returncode == 0


def test_generated_script_does_not_depend_on_easybfe(leg: Path) -> None:
    """The leg must run in an AMBER-only environment. See CLAUDE.md.

    Comments and the ``EASYBFE_*`` environment variables are fine; what must
    never appear is a call out to easybfe or to a python interpreter.
    """
    code = "\n".join(
        line for line in (leg / "run.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    code = re.sub(r"EASYBFE_[A-Z0-9_]*", "", code)

    assert "easybfe" not in code.lower(), "leg script must not invoke easybfe"
    assert "python" not in code.lower(), "leg script must not invoke python"


def test_only_one_leg_script_is_generated(leg: Path) -> None:
    """The phases are selected by argument, not by three near-identical scripts."""
    assert sorted(p.name for p in leg.glob("run*.sh")) == ["run.sh"]


def test_list_reports_every_stage(leg: Path, stub_env: dict) -> None:
    proc = run_leg(leg, stub_env, "--list")
    assert proc.returncode == 0
    listed = [line.split("\t")[0] for line in proc.stdout.strip().splitlines()]
    assert listed == ["01.em", "02.heat", "03.pres", "04.pre_prod", "05.prod"]


def test_unknown_stage_is_rejected(leg: Path, stub_env: dict) -> None:
    proc = run_leg(leg, stub_env, "--from", "99.nope")
    assert proc.returncode == 2
    assert "Unknown stage" in proc.stderr


def test_until_runs_the_preprod_phase_only(leg: Path, stub_env: dict) -> None:
    proc = run_leg(leg, stub_env, "--until", "04.pre_prod")

    assert proc.returncode == 0
    assert (leg / "04.pre_prod.done.tag").is_file()
    assert (leg / "preprod.done.tag").is_file()
    assert not (leg / "05.prod.done.tag").exists()
    assert not (leg / "done.tag").exists(), "the leg is not finished yet"


def test_from_completes_the_leg_and_skips_finished_stages(leg: Path, stub_env: dict) -> None:
    run_leg(leg, stub_env, "--until", "04.pre_prod")
    proc = run_leg(leg, stub_env, "--from", "05.prod")

    assert proc.returncode == 0
    assert (leg / "done.tag").is_file()
    assert "Running 05.prod" in proc.stdout


def test_rerun_skips_completed_stages(leg: Path, stub_env: dict) -> None:
    run_leg(leg, stub_env)
    proc = run_leg(leg, stub_env, "--force")

    assert proc.returncode == 0
    assert "Skipping 01.em (already done)" in proc.stdout


def test_failure_records_status_and_surfaces_the_cause(leg: Path, stub_env: dict) -> None:
    env = dict(stub_env, STUB_FAIL_STAGE="05.prod")
    proc = run_leg(leg, env)

    assert proc.returncode != 0
    assert (leg / "error.tag").is_file()
    assert not (leg / "done.tag").exists()
    # The root cause has to reach whoever launched the script, not only the log.
    assert "cuStreamCreate return value: 709" in proc.stdout

    status = json.loads((leg / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["stage"] == "05.prod"
    assert status["exit_code"] == 1
    assert status["stages"]["04.pre_prod"] == "done"
    assert status["stages"]["05.prod"] == "failed"
    assert "709" in status["error_excerpt"]


def test_error_excerpt_survives_nul_bytes_in_the_log(leg: Path, stub_env: dict) -> None:
    """pmemd emits NULs; grep would otherwise report only "Binary file ... matches",
    which is what a real failed leg reported instead of its actual error."""
    env = dict(stub_env, STUB_FAIL_STAGE="05.prod", STUB_BINARY_LOG="1")
    run_leg(leg, env)

    excerpt = json.loads((leg / "status.json").read_text())["error_excerpt"]
    assert "Binary file" not in excerpt
    assert "709" in excerpt


def test_error_tag_blocks_a_plain_rerun_but_not_force(leg: Path, stub_env: dict) -> None:
    """Manual re-runs must not silently retry a failed leg; --force is explicit."""
    run_leg(leg, dict(stub_env, STUB_FAIL_STAGE="05.prod"))
    assert (leg / "error.tag").is_file()

    blocked = run_leg(leg, stub_env)
    assert "error.tag" in blocked.stdout
    assert not (leg / "done.tag").exists()

    forced = run_leg(leg, stub_env, "--force")
    assert forced.returncode == 0
    assert (leg / "done.tag").is_file()


def test_completed_status_after_a_full_run(leg: Path, stub_env: dict) -> None:
    proc = run_leg(leg, stub_env)

    assert proc.returncode == 0
    status = json.loads((leg / "status.json").read_text())
    assert status["state"] == "completed"
    assert set(status["stages"].values()) == {"done"}
    assert status["error_excerpt"] is None
