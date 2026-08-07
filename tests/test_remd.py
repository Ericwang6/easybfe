"""Tests for :func:`easybfe.analysis.remd.parse_remlog`."""

from __future__ import annotations

from pathlib import Path

import pytest

from easybfe.analysis.remd import parse_remlog


HEADER = """# Replica Exchange log file
# numexchg is          2
# REMD filenames:
#   remlog= 05.prod.log
#   remtype= rem.type
# Rep#, Neibr#, Temp0, PotE(x_1), PotE(x_2), left_fe, right_fe, Success, Success rate (i,i+1)
"""

# Four replicas, two exchange attempts. Odd exchanges pair (2,3) and (4,1);
# even exchanges pair (1,2) and (3,4). Pair (4,1) is the wrap-around row and is
# never a real exchange.
REMLOG = HEADER + """# exchange        1
     1     4    298.15 -20684.21 -20700.10   -430.31      0.00    F        0.00
     2     3    298.15 -20854.47 -20914.32      0.00    -10.37    T        1.00
     3     2    298.15 -20903.99 -20844.10     10.32      0.00    T        0.00
     4     1    298.15 -20485.36 -20253.90      0.00  Infinity    F        0.00
# exchange        2
     1     2    298.15 -20749.23 -20908.43   -430.31     -9.13    T        1.00
     2     1    298.15 -20901.17 -20740.11      7.26    -10.37    T        1.00
     3     4    298.15 -20898.47 -20902.28     10.32     -9.74    F        0.00
     4     3    298.15 -20892.67 -20888.74      9.61  Infinity    F        0.00
"""


def _write(tmp_path: Path, text: str) -> Path:
    log_file = tmp_path / "05.prod.log"
    log_file.write_text(text)
    return log_file


def test_parse_remlog_acceptance_rates(tmp_path: Path) -> None:
    """Rates are per neighbour pair; the wrap-around row is not a pair."""
    result = parse_remlog(_write(tmp_path, REMLOG))

    assert result["n_replicas"] == 4
    assert result["n_exchanges"] == 2
    # pair (1,2) accepted once of one attempt, (2,3) once of one, (3,4) zero of one.
    assert result["exchange_rate_per_pair"] == [1.0, 1.0, 0.0]
    assert result["exchange_attempts_per_pair"] == [1, 1, 1]
    assert result["exchange_rate"] == pytest.approx(2 / 3)
    assert result["exchange_rate_min"] == 0.0
    assert result["exchange_rate_max"] == 1.0


def test_parse_remlog_handles_merged_columns(tmp_path: Path) -> None:
    """Fixed-width columns can run together; parsing works from both ends."""
    text = HEADER + """# exchange        1
     1     2    298.15-117604.09**********     -1.00      0.00    T        1.00
     2     1    298.15-117610.00 -117604.09     1.00      0.00    T        0.00
"""
    result = parse_remlog(_write(tmp_path, text))

    assert result["n_replicas"] == 2
    assert result["exchange_rate_per_pair"] == [1.0]


def test_parse_remlog_missing_file(tmp_path: Path) -> None:
    """A stage without replica exchange has no remlog to report."""
    assert parse_remlog(tmp_path / "04.pre_prod.log") is None


def test_parse_remlog_without_exchange_rows(tmp_path: Path) -> None:
    """A header-only log (e.g. a stage killed at startup) yields nothing."""
    assert parse_remlog(_write(tmp_path, HEADER)) is None
