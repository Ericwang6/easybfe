import os
import logging
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import alchemlyb
from alchemlyb.estimators import MBAR
from alchemlyb.parsing.amber import extract_u_nk
from alchemlyb.convergence import forward_backward_convergence
from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix
from matplotlib.axes import Axes


logger = logging.getLogger(__name__)


_COLOR_FORWARD = "#1B4965"
_COLOR_BACKWARD = "#C44536"
_COLOR_FORWARD_FILL = "#62B6CB"
_COLOR_BACKWARD_FILL = "#F4A261"
_COLOR_FINAL_BAND = "#B8B8D1"


def plot_convergence(
    dataframe: pd.DataFrame,
    units: str | None = None,
    final_error: float | None = None,
    ax: Axes | None = None,
    *,
    x_column: str | None = "data_fraction",
) -> Axes:
    """Plot forward and backward cumulative convergence with styled defaults."""
    forward = dataframe["Forward"].to_numpy(dtype=float)
    if "Forward_Error" in dataframe:
        forward_error = dataframe["Forward_Error"].to_numpy(dtype=float)
    else:
        forward_error = np.zeros(len(forward))
    backward = dataframe["Backward"].to_numpy(dtype=float)
    if "Backward_Error" in dataframe:
        backward_error = dataframe["Backward_Error"].to_numpy(dtype=float)
    else:
        backward_error = np.zeros(len(backward))

    n = len(forward)
    if x_column is not None and x_column in dataframe.columns:
        f_ts = r_ts = dataframe[x_column].to_numpy(dtype=float)
    else:
        f_ts = np.linspace(0, 1, n + 1)[1:]
        r_ts = np.linspace(0, 1, n + 1)[1:]

    if final_error is None:
        final_error = float(backward_error[-1])

    if units is None:
        units = dataframe.attrs.get("energy_unit", "kcal/mol")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6.5), dpi=120)
    ax.set_facecolor("#FFFFFF")

    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#4A4A4A")
        ax.spines[spine].set_linewidth(0.8)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, color="#888888")
    ax.set_axisbelow(True)

    if np.isfinite(backward[-1]) and np.isfinite(final_error):
        ax.axhspan(
            backward[-1] - final_error,
            backward[-1] + final_error,
            facecolor=_COLOR_FINAL_BAND,
            edgecolor="none",
            alpha=0.35,
            zorder=1,
            label="_nolegend_",
        )

    m = np.isfinite(forward) & np.isfinite(forward_error)
    if m.any():
        ax.fill_between(
            f_ts[m],
            forward[m] - forward_error[m],
            forward[m] + forward_error[m],
            color=_COLOR_FORWARD_FILL,
            alpha=0.35,
            zorder=2,
            linewidth=0,
        )
    m = np.isfinite(backward) & np.isfinite(backward_error)
    if m.any():
        ax.fill_between(
            r_ts[m],
            backward[m] - backward_error[m],
            backward[m] + backward_error[m],
            color=_COLOR_BACKWARD_FILL,
            alpha=0.35,
            zorder=3,
            linewidth=0,
        )

    ax.errorbar(
        f_ts,
        forward,
        yerr=forward_error,
        fmt="none",
        ecolor=_COLOR_FORWARD,
        elinewidth=1.3,
        capsize=3.0,
        alpha=0.95,
        zorder=4,
    )
    ax.errorbar(
        r_ts,
        backward,
        yerr=backward_error,
        fmt="none",
        ecolor=_COLOR_BACKWARD,
        elinewidth=1.3,
        capsize=3.0,
        alpha=0.95,
        zorder=5,
    )

    ax.plot(
        f_ts,
        forward,
        color=_COLOR_FORWARD,
        lw=2.2,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.8,
        markeredgecolor=_COLOR_FORWARD,
        zorder=6,
        label="Forward",
    )
    ax.plot(
        r_ts,
        backward,
        color=_COLOR_BACKWARD,
        lw=2.2,
        marker="s",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=1.8,
        markeredgecolor=_COLOR_BACKWARD,
        zorder=7,
        label="Reverse",
    )

    ax.set_xlabel("Fraction of simulation time", fontsize=13, color="#222222")
    ax.set_ylabel(rf"$\Delta G$ ({units})", fontsize=13, color="#222222")
    ax.set_xlim(0.05, 1.05)
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    ax.tick_params(axis="both", labelsize=12, colors="#333333")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="best",
        frameon=True,
        framealpha=0.92,
        edgecolor="#DDDDDD",
        fontsize=11,
    )
    fig = ax.figure
    fig.patch.set_facecolor("#FFFFFF")
    fig.tight_layout()
    return ax


@dataclass
class MBARResult:
    dg: float
    dg_std: float
    convergence: pd.DataFrame
    overlap: np.ndarray


def run_mbar(
    dirname: os.PathLike,
    prefix: str = 'prod',
    temperature: float = 298.15
):
    """
    Run MBAR (Multistate Bennett Acceptance Ratio) analysis to compute free energy differences 
    and perform convergence analysis based on simulation data.
    
    Parameters
    ----------
    dirname : os.PathLike
        Path to the directory containing simulation output files. The directory is expected 
        to have subdirectories named `lambdaX` (where X is an integer) containing production 
        output files (`lambdaX/prod/prod.out`).
    temperature : float, optional
        Temperature in Kelvin at which the simulations were performed. Default is 298.15 K.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages. If None, a default logger will be initialized.
    prefix : str, optional
        Name of the production run subdirectory and output file prefix. For each lambda window,
        the function expects output files at `lambdaX/{prod_name}/{prod_name}.out`. Default is 'prod'.
    
    Returns
    -------
    dg : float
        Free energy difference (Delta G) in kcal/mol computed using MBAR.
    dg_std : float
        Standard deviation of the free energy difference in kcal/mol.
    
    Notes
    -----
    - This function extracts reduced potentials from simulation output files, evaluates free 
      energy differences using MBAR, performs convergence analysis, and generates plots for 
      overlap matrix and convergence analysis.
    - The convergence analysis results are saved as a CSV file (`convergence.csv`) and a plot 
      (`convergence.png`) in the specified directory.
    - The overlap matrix plot is saved as `overlap.png` in the specified directory.
    
    Examples
    --------
    >>> dirname = "/path/to/simulation/output"
    >>> temperature = 300.0
    >>> dg, dg_std = run_mbar(dirname, temperature)
    >>> print(f"Free energy difference: {dg:.2f} ± {dg_std:.2f} kcal/mol")
    """
    
    dirname = Path(dirname).resolve()
    kBT = 8.314 * temperature / 1000 / 4.184

    logger.info("Extracting data from output...")
    u_nks = []
    num_lambda = len(list(dirname.glob('lambda*')))
    for i in tqdm(range(num_lambda), leave=True):
        out = str(dirname / f"lambda{i}/{prefix}/{prefix}.out")
        u_nks.append(extract_u_nk(out, T=temperature))
    
    # evaluate free energy with MBAR
    logger.info("Running MBAR estimator...")
    mbarEstimator = MBAR()
    mbarEstimator.fit(alchemlyb.concat(u_nks))
    dg = mbarEstimator.delta_f_.iloc[0, -1] * kBT
    dg_std = mbarEstimator.d_delta_f_.iloc[0, -1] * kBT
    
    # convergence analysis
    logger.info("Running convergence analysis...")
    conv_df = forward_backward_convergence(u_nks, "MBAR")
    for key in ['Forward', 'Forward_Error', 'Backward', 'Backward_Error']:
        conv_df[key] *= kBT
    conv_df.to_csv(dirname / "convergence.csv", index=None)
    conv_ax = plot_convergence(conv_df)
    conv_ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
    conv_ax.set_title(
        f"Convergence Analysis - {dirname.name.capitalize()}",
        fontsize=14,
        fontweight="semibold",
        pad=12,
    )
    conv_ax.figure.tight_layout(rect=(0, 0, 1, 0.97))
    conv_ax.figure.savefig(str(dirname /"convergence.png"), dpi=300)

    # overlap matrix
    logger.info("Plotting overlap matrix...")
    overlap_ax = plot_mbar_overlap_matrix(mbarEstimator.overlap_matrix)
    overlap_ax.figure.savefig(str(dirname /"overlap.png"), dpi=300)

    return MBARResult(dg, dg_std, conv_df, mbarEstimator.overlap_matrix)

