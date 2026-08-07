"""The pre-production estimate must not run in the pipeline's own process.

pymbar solves through JAX, which takes ~75% of the first visible GPU on its
first computation and holds it until the process exits. The early-stop estimate
is the one analysis with MD still to come, so it has to happen in a child
process the OS can reclaim -- otherwise every production rank afterwards dies in
cudaMalloc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from easybfe.abfe.piepline import ABFE


class _Runner:
    """``ABFE`` reduced to the analysis plumbing under test."""

    def __init__(self, abfe_dir: Path):
        self.abfe_dir = abfe_dir

    _analyze_out_of_process = ABFE._analyze_out_of_process


@pytest.fixture
def abfe_dir(tmp_path: Path) -> Path:
    return tmp_path


def test_analysis_runs_in_a_child_process(abfe_dir: Path, monkeypatch) -> None:
    """The child must be a separate interpreter, not an in-process call."""
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        # Stand in for analyze_abfe: the real one writes result.json.
        (abfe_dir / "result.json").write_text(json.dumps({"total": -12.5}))

        class _Proc:
            returncode = 0

        return _Proc()

    monkeypatch.setattr("subprocess.run", fake_run)
    result = _Runner(abfe_dir)._analyze_out_of_process({"prod_prefix": "04.pre_prod"})

    assert result == {"total": -12.5}
    assert len(calls) == 1
    assert calls[0][0] == sys.executable, "must spawn a fresh interpreter"
    # The kwargs have to survive the process boundary intact.
    payload = json.loads(calls[0][-1])
    assert payload["prod_prefix"] == "04.pre_prod"
    assert payload["directory"] == str(abfe_dir)


def test_child_failure_is_raised_not_swallowed(abfe_dir: Path, monkeypatch) -> None:
    def fake_run(cmd, *args, **kwargs):
        class _Proc:
            returncode = 1

        return _Proc()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 1"):
        _Runner(abfe_dir)._analyze_out_of_process({"prod_prefix": "04.pre_prod"})


def test_missing_result_json_is_not_an_error(abfe_dir: Path, monkeypatch) -> None:
    """Incomplete legs mean no result; that is reported as {}, not a crash."""

    def fake_run(cmd, *args, **kwargs):
        class _Proc:
            returncode = 0

        return _Proc()

    monkeypatch.setattr("subprocess.run", fake_run)

    assert _Runner(abfe_dir)._analyze_out_of_process({}) == {}


def test_it_really_works_end_to_end(abfe_dir: Path) -> None:
    """No mocks: spawn the real child and confirm the round trip.

    Uses a stub module on the child's path so the test stays fast and does not
    need real MD output -- what is being checked is the process boundary and the
    JSON hand-off, not MBAR itself.
    """
    stub = abfe_dir / "easybfe_stub"
    stub.mkdir()
    # A fake analyze_abfe that just records what it was handed.
    (stub / "sitecustomize.py").write_text(
        "import json, sys\n"
        "import easybfe.analysis.abfe as m\n"
        "def _fake(directory, **kw):\n"
        "    import pathlib\n"
        "    (pathlib.Path(directory) / 'result.json').write_text(\n"
        "        json.dumps({'total': -1.0, 'seen': sorted(kw)}))\n"
        "m.analyze_abfe = _fake\n"
    )
    import os
    import subprocess

    env = dict(os.environ, PYTHONPATH=f"{stub}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    real_run = subprocess.run

    def run_with_stub(cmd, *args, **kwargs):
        return real_run(cmd, *args, env=env, **kwargs)

    runner = _Runner(abfe_dir)
    import unittest.mock as mock

    with mock.patch("subprocess.run", run_with_stub):
        result = runner._analyze_out_of_process({"prod_prefix": "04.pre_prod"})

    assert result["total"] == -1.0
    assert "prod_prefix" in result["seen"]
