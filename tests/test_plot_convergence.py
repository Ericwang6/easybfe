"""Tests for convergence plotting in :mod:`easybfe.analysis.mbar`."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from easybfe.analysis.mbar import plot_convergence

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
