"""One failing leg must not stop the others.

A single submission should get as far as it can, so that only the legs that
actually failed have to be re-run.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import pytest

from easybfe.abfe.piepline import ABFE


LEGS = ("solvent", "complex", "restraint")


class _Runner:
    """Stands in for ``ABFE`` with only the leg-running machinery kept."""

    def __init__(self, abfe_dir: Path, failing: set[str]):
        self.abfe_dir = abfe_dir
        self.failing = failing
        self.attempted: list[tuple[str, tuple]] = []

    def _run_script(self, directory, script="run.sh", args=None, done_tag="done.tag"):
        leg = Path(directory).name
        self.attempted.append((leg, tuple(args or ())))
        return leg not in self.failing

    # Reuse the real reporting/status helpers under test.
    _report_failed_legs = ABFE._report_failed_legs
    _leg_status = ABFE._leg_status
    run_abfe = ABFE.run_abfe


@pytest.fixture
def abfe_dir(tmp_path: Path) -> Path:
    for leg in LEGS:
        (tmp_path / leg).mkdir()
    return tmp_path


def test_all_legs_are_attempted_when_one_fails(abfe_dir: Path) -> None:
    runner = _Runner(abfe_dir, failing={"solvent"})

    results = runner.run_abfe()

    assert [leg for leg, _ in runner.attempted] == list(LEGS)
    assert results == {"solvent": False, "complex": True, "restraint": True}


@contextmanager
def collect_errors():
    """Capture ERROR records from the pipeline logger.

    A plain ``caplog`` is not enough here: other tests in the suite run the CLI,
    which attaches its own handlers to the ``easybfe`` logger, so what reaches
    the root logger depends on test order.
    """
    records: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                records.append(record.getMessage())

    logger = logging.getLogger("easybfe.abfe.piepline")
    handler = _Collector()
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_failed_legs_are_reported_with_a_rerun_command(abfe_dir: Path) -> None:
    runner = _Runner(abfe_dir, failing={"complex"})

    with collect_errors() as records:
        runner.run_abfe()

    messages = "\n".join(records)
    assert "complex" in messages
    assert "run.sh --force" in messages


def test_no_error_logged_when_every_leg_succeeds(abfe_dir: Path) -> None:
    runner = _Runner(abfe_dir, failing=set())

    with collect_errors() as records:
        runner.run_abfe()

    assert records == []


def test_leg_status_is_read_from_status_json(abfe_dir: Path) -> None:
    payload = {"leg": "solvent", "state": "failed", "stage": "05.prod"}
    (abfe_dir / "solvent" / "status.json").write_text(json.dumps(payload))
    runner = _Runner(abfe_dir, failing=set())

    assert runner._leg_status(abfe_dir / "solvent") == payload
    # A leg that never started has no status file; that is not an error.
    assert runner._leg_status(abfe_dir / "complex") == {}
