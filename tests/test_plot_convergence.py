"""Tests for convergence plotting in :mod:`easybfe.analysis.mbar`."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from easybfe.analysis.mbar import plot_block_average, plot_convergence

_DATA_DIR = Path(__file__).resolve().parent / "data"
CONVERGENCE_CSV = _DATA_DIR / "convergence.csv"
CONVERGENCE_PNG = Path(__file__).resolve().parent / "convergence_styled.png"


def test_plot_convergence_renders_convergence_csv() -> None:
    """Load convergence data and write a PNG under tests/."""
    pytest.importorskip("matplotlib")
    df = pd.read_csv(CONVERGENCE_CSV)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    plot_convergence(df, units="kcal/mol", ax=ax)
    fig.savefig(CONVERGENCE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert CONVERGENCE_PNG.is_file() and CONVERGENCE_PNG.stat().st_size > 0


def test_plot_convergence_matches_expected_columns() -> None:
    df = pd.read_csv(CONVERGENCE_CSV)
    assert set(df.columns) >= {
        "Forward",
        "Forward_Error",
        "Backward",
        "Backward_Error",
        "data_fraction",
    }
    assert len(df) == 10
    assert np.allclose(df["data_fraction"].iloc[-1], 1.0)


def test_plot_block_average_renders_block_columns() -> None:
    pytest.importorskip("matplotlib")
    df = pd.read_csv(CONVERGENCE_CSV)
    block = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    df["Block_Average"] = block
    df["Block_Average_Error"] = df["Forward_Error"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    plot_block_average(df, units="kcal/mol", ax=ax)
    plt.close(fig)
    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["Average", "Block average"]

    mean_line = next(line for line in ax.lines if line.get_linestyle() == "--")
    block_mean = float(np.mean(block))
    assert np.allclose(mean_line.get_ydata(), block_mean)
    assert len(ax.patches) == 1
