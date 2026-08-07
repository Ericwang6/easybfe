"""Tests for the diagnostics written into ABFE ``result.json``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easybfe.analysis.abfe import (
    _block_average_summary,
    _convergence_summary,
    _infer_early_stop,
    _overlap_summary,
)


def _conv_df(forward: list[float], backward: list[float]) -> pd.DataFrame:
    n = len(forward)
    df = pd.DataFrame(
        {
            "Forward": forward,
            "Forward_Error": [0.5] * n,
            "Backward": backward,
            "Backward_Error": [0.5] * n,
            "data_fraction": np.linspace(1 / n, 1.0, n),
            "Block_Average": forward,
            "Block_Average_Error": [0.5] * n,
        }
    )
    from easybfe.analysis.mbar import annotate_convergence_to_conv_df

    annotate_convergence_to_conv_df(df)
    return df


def test_convergence_summary_flags_a_converged_run() -> None:
    """Both series within the final error over the second half is converged."""
    values = [-12.0, -10.5, -10.2, -10.1, -10.0, -10.0, -10.1, -10.0, -10.0, -10.0]
    summary = _convergence_summary(_conv_df(values, values))

    assert summary["is_converged"] is True
    assert summary["final_forward"] == pytest.approx(-10.0)
    assert len(summary["forward"]) == len(values)
    assert summary["data_fraction"][-1] == pytest.approx(1.0)


def test_convergence_summary_flags_a_drifting_run() -> None:
    """A drift larger than the final error late in the run is not converged."""
    forward = [-12.0, -11.5, -11.0, -10.5, -8.0, -9.0, -9.5, -10.0, -10.0, -10.0]
    summary = _convergence_summary(_conv_df(forward, forward))

    assert summary["is_converged"] is False
    assert summary["converged"][4] is False


def test_block_average_summary_reports_spread() -> None:
    values = [-11.0, -9.0, -11.0, -9.0, -11.0, -9.0, -11.0, -9.0, -11.0, -9.0]
    summary = _block_average_summary(_conv_df(values, values))

    assert summary["mean"] == pytest.approx(-10.0)
    assert summary["std"] == pytest.approx(1.0)


def test_overlap_summary_uses_the_first_off_diagonal() -> None:
    matrix = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.3, 0.4, 0.3],
            [0.1, 0.3, 0.6],
        ]
    )
    summary = _overlap_summary(matrix)

    assert summary["n_states"] == 3
    assert summary["adjacent"] == pytest.approx([0.3, 0.3])
    assert summary["min_adjacent"] == pytest.approx(0.3)


def test_overlap_summary_of_a_degenerate_matrix() -> None:
    assert _overlap_summary(np.array([[1.0]]))["min_adjacent"] is None


def _make_legs(tmp_path: Path, tags: dict[str, list[str]]) -> Path:
    for leg, leg_tags in tags.items():
        leg_dir = tmp_path / leg
        leg_dir.mkdir()
        for tag in leg_tags:
            (leg_dir / tag).touch()
    return tmp_path


def test_infer_early_stop_true_when_production_never_ran(tmp_path: Path) -> None:
    wdir = _make_legs(
        tmp_path, {leg: ["preprod.done.tag"] for leg in ("complex", "solvent", "restraint")}
    )
    assert _infer_early_stop(wdir) is True


def test_infer_early_stop_false_after_a_full_run(tmp_path: Path) -> None:
    wdir = _make_legs(
        tmp_path,
        {leg: ["preprod.done.tag", "done.tag"] for leg in ("complex", "solvent", "restraint")},
    )
    assert _infer_early_stop(wdir) is False


def test_infer_early_stop_false_for_a_single_phase_run(tmp_path: Path) -> None:
    """``run.sh`` (no early-stop phases) only leaves ``done.tag``."""
    wdir = _make_legs(
        tmp_path, {leg: ["done.tag"] for leg in ("complex", "solvent", "restraint")}
    )
    assert _infer_early_stop(wdir) is False
