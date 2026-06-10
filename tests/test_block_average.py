"""Tests for :func:`easybfe.analysis.mbar.block_average`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from easybfe.analysis.mbar import block_average


def _make_u_nk(n_frames: int) -> pd.DataFrame:
    """Minimal single-lambda-window u_nk for block slicing tests."""
    index = pd.MultiIndex.from_arrays(
        [np.arange(n_frames), np.zeros(n_frames)],
        names=["time", "fep-lambda"],
    )
    columns = pd.MultiIndex.from_product(
        [[0.0, 1.0], [0.0, 1.0]],
        names=["fep-lambda", "fep-lambda"],
    )
    data = np.random.default_rng(0).random((n_frames, 4))
    df = pd.DataFrame(data, index=index, columns=columns)
    df.attrs = {"temperature": 300, "energy_unit": "kT"}
    return df


def _mock_estimator_result() -> MagicMock:
    mock_result = MagicMock()
    mock_result.delta_f_ = pd.DataFrame([[0.0, 1.5]], index=[0.0], columns=[0.0, 1.0])
    mock_result.d_delta_f_ = pd.DataFrame([[0.0, 0.1]], index=[0.0], columns=[0.0, 1.0])
    return mock_result


def test_block_average_returns_num_blocks() -> None:
    """``num`` blocks should all be estimated, including the final segment."""
    u_nk = _make_u_nk(100)
    mock_cls = MagicMock()
    mock_cls.return_value.fit.return_value = _mock_estimator_result()

    with patch.dict("easybfe.analysis.mbar._ESTIMATORS", {"MBAR": mock_cls}):
        df = block_average([u_nk], estimator="MBAR", num=10)

    assert df.shape == (10, 2)
    assert list(df.columns) == ["FE", "FE_Error"]
    assert mock_cls.return_value.fit.call_count == 10


def test_block_average_last_block_includes_remainder() -> None:
    """The final block should extend to ``len(data)``, not stop at integer division."""
    u_nk = _make_u_nk(105)
    slices: list[tuple[int, int]] = []

    def capture_fit(sample_df: pd.DataFrame) -> MagicMock:
        times = sample_df.index.get_level_values("time")
        slices.append((int(times.min()), int(times.max()) + 1))
        return _mock_estimator_result()

    mock_cls = MagicMock()
    mock_cls.return_value.fit.side_effect = capture_fit

    with patch("easybfe.analysis.mbar.alchemlyb.concat", side_effect=lambda parts: parts[0]):
        with patch.dict("easybfe.analysis.mbar._ESTIMATORS", {"MBAR": mock_cls}):
            block_average([u_nk], estimator="MBAR", num=10)

    assert slices[-1] == (90, 105)
