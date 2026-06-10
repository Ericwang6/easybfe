"""Tests for :func:`easybfe.analysis.mbar.decorrelate_u_nks`."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from easybfe.analysis.mbar import decorrelate_u_nks


def _make_u_nk(n_frames: int) -> pd.DataFrame:
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


def test_decorrelate_u_nks_calls_per_lambda_window() -> None:
    u_nks = [_make_u_nk(50), _make_u_nk(60)]
    subsampled = [_make_u_nk(10), _make_u_nk(12)]

    with patch(
        "easybfe.analysis.mbar.decorrelate_u_nk",
        side_effect=subsampled,
    ) as mock_decorrelate:
        result = decorrelate_u_nks(
            u_nks, method="dE", remove_burnin=True, n_jobs=1
        )

    assert mock_decorrelate.call_count == 2
    assert len(result) == 2
    assert len(result[0]) == 10
    assert len(result[1]) == 12
