import os
import logging
from typing import Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib



def _silence_pymbar_import_messages() -> None:
    """Suppress noisy pymbar messages emitted at import time."""
    suppressed = (
        "Warning on use of the timeseries module:",
        "****** PyMBAR will use 64-bit JAX! *******",
    )

    class _SuppressMessage(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(text in msg for text in suppressed)

    filt = _SuppressMessage()
    for name in ("pymbar.timeseries", "pymbar.mbar_solvers"):
        logging.getLogger(name).addFilter(filt)


_silence_pymbar_import_messages()

import alchemlyb
from alchemlyb.estimators import BAR, FEP_ESTIMATORS, MBAR, TI, TI_ESTIMATORS
from alchemlyb.parsing.amber import extract_u_nk
from alchemlyb.convergence import forward_backward_convergence
from alchemlyb.preprocessing import decorrelate_u_nk
from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix
from matplotlib.axes import Axes


logger = logging.getLogger(__name__)


_COLOR_FORWARD = "#1B4965"
_COLOR_BACKWARD = "#C44536"
_COLOR_BLOCK = "#2D6A4F"
_COLOR_FORWARD_FILL = "#62B6CB"
_COLOR_BACKWARD_FILL = "#F4A261"
_COLOR_FINAL_BAND = "#B8B8D1"


def _style_convergence_axis(ax: Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#4A4A4A")
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, color="#888888")
    ax.set_axisbelow(True)


def _convergence_x_values(dataframe: pd.DataFrame, n: int, x_column: str | None) -> np.ndarray:
    if x_column is not None and x_column in dataframe.columns:
        return dataframe[x_column].to_numpy(dtype=float)
    return np.linspace(0, 1, n + 1)[1:]


def _finalize_convergence_figure(ax: Axes, *, xlabel: str, ylabel: str) -> Axes:
    ax.set_xlabel(xlabel, fontsize=13, color="#222222")
    ax.set_ylabel(ylabel, fontsize=13, color="#222222")
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
    f_ts = _convergence_x_values(dataframe, n, x_column)
    r_ts = f_ts

    if final_error is None:
        final_error = float(backward_error[-1])

    if units is None:
        units = dataframe.attrs.get("energy_unit", "kcal/mol")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6.5), dpi=120)
    _style_convergence_axis(ax)

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

    return _finalize_convergence_figure(
        ax,
        xlabel="Fraction of simulation time",
        ylabel=rf"$\Delta G$ ({units})",
    )


def _resolve_block_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    if "Block_Average" in dataframe:
        err_col = "Block_Average_Error" if "Block_Average_Error" in dataframe else None
        return "Block_Average", err_col
    if "FE" in dataframe:
        err_col = "FE_Error" if "FE_Error" in dataframe else None
        return "FE", err_col
    raise ValueError(
        "dataframe must contain Block_Average or FE columns for block averaging."
    )


def plot_block_average(
    dataframe: pd.DataFrame,
    units: str | None = None,
    final_error: float | None = None,
    ax: Axes | None = None,
    *,
    x_column: str | None = "data_fraction",
) -> Axes:
    """Plot block-averaged free energy estimates with styled defaults.

    Per-block estimates are drawn with their estimator uncertainties.
    A horizontal band spanning the full simulation time shows the mean
    block estimate plus/minus the standard deviation across blocks, following
    :func:`alchemlyb.visualisation.plot_block_average`.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Must contain ``Block_Average`` (or ``FE``) and optionally
        ``Block_Average_Error`` (or ``FE_Error``).
    units : str, optional
        Energy unit for the y-axis label. Defaults to
        ``dataframe.attrs['energy_unit']``.
    final_error : float, optional
        Half-width of the mean band in ``units``. Defaults to the standard
        deviation of the block estimates.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when ``None``.
    x_column : str, optional
        Column used for block x positions. Defaults to ``data_fraction``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the block-average plot.
    """
    fe_col, fe_err_col = _resolve_block_columns(dataframe)
    block = dataframe[fe_col].to_numpy(dtype=float)
    if fe_err_col is not None:
        block_error = dataframe[fe_err_col].to_numpy(dtype=float)
    else:
        block_error = np.zeros(len(block))

    b_ts = _convergence_x_values(dataframe, len(block), x_column)
    block_mean = float(np.mean(block))

    if final_error is None:
        final_error = float(np.std(block))

    if units is None:
        units = dataframe.attrs.get("energy_unit", "kcal/mol")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6.5), dpi=120)
    _style_convergence_axis(ax)

    if final_error is not None and np.isfinite(final_error):
        ax.axhspan(
            block_mean - final_error,
            block_mean + final_error,
            facecolor=_COLOR_FINAL_BAND,
            edgecolor="none",
            alpha=0.35,
            zorder=1,
            label="_nolegend_",
        )

    ax.axhline(
        block_mean,
        color=_COLOR_BLOCK,
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
        zorder=2,
        label="Average",
    )

    ax.errorbar(
        b_ts,
        block,
        yerr=block_error,
        fmt="none",
        ecolor=_COLOR_BLOCK,
        elinewidth=1.3,
        capsize=3.0,
        alpha=0.95,
        zorder=3,
    )
    ax.plot(
        b_ts,
        block,
        color=_COLOR_BLOCK,
        lw=2.2,
        marker="D",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        markeredgecolor=_COLOR_BLOCK,
        zorder=4,
        label="Block average",
    )

    return _finalize_convergence_figure(
        ax,
        xlabel="Fraction of simulation time",
        ylabel=rf"$\Delta G$ ({units})",
    )


def save_convergence_plots(
    conv_df: pd.DataFrame,
    convergence_png: os.PathLike,
    block_average_png: os.PathLike,
    *,
    title: str,
    ylabel: str,
    block_average_title: str | None = None,
) -> None:
    """Save forward/backward and block-average convergence plots."""
    conv_ax = plot_convergence(conv_df)
    conv_ax.set_ylabel(ylabel)
    conv_ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)
    conv_ax.figure.tight_layout(rect=(0, 0, 1, 0.97))
    conv_ax.figure.savefig(str(convergence_png), dpi=300)
    plt.close(conv_ax.figure)

    if "Block_Average" not in conv_df:
        return

    block_ax = plot_block_average(conv_df)
    block_ax.set_ylabel(ylabel)
    block_ax.set_title(
        block_average_title or title.replace("Convergence", "Block Average"),
        fontsize=14,
        fontweight="semibold",
        pad=12,
    )
    block_ax.figure.tight_layout(rect=(0, 0, 1, 0.97))
    block_ax.figure.savefig(str(block_average_png), dpi=300)
    plt.close(block_ax.figure)


_ESTIMATORS = {"BAR": BAR, "TI": TI, "MBAR": MBAR}


def _decorrelate_u_nk_one(
    u_nk: pd.DataFrame,
    method: str,
    remove_burnin: bool,
) -> pd.DataFrame:
    """Decorrelate one lambda-window ``u_nk`` DataFrame."""
    return decorrelate_u_nk(
        u_nk,
        method=method,
        remove_burnin=remove_burnin,
    )


def decorrelate_u_nks(
    u_nks: List[pd.DataFrame],
    method: str = "dE",
    remove_burnin: bool = True,
    n_jobs: int = -1,
) -> List[pd.DataFrame]:
    """Decorrelate each lambda-window ``u_nk`` DataFrame.

    Parameters
    ----------
    u_nks : list of pandas.DataFrame
        One ``u_nk`` per lambda window.
    method : {'all', 'dE'}
        Series used for autocorrelation analysis; see
        :func:`alchemlyb.preprocessing.decorrelate_u_nk`.
    remove_burnin : bool
        If ``True``, detect equilibration before subsampling.
    n_jobs : int, optional
        Number of parallel workers. ``-1`` uses all available workers
        (see :func:`joblib.Parallel`). Default is ``-1``.

    Returns
    -------
    list of pandas.DataFrame
        Decorrelated ``u_nk`` DataFrames, one per lambda window.
    """
    decorrelated = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_decorrelate_u_nk_one)(u_nk, method, remove_burnin)
        for u_nk in u_nks
    )
    for i, (u_nk, u_nk_sub) in enumerate(zip(u_nks, decorrelated)):
        logger.info(
            "Lambda %d: %d -> %d frames after decorrelation.",
            i,
            len(u_nk),
            len(u_nk_sub),
        )
    return decorrelated


def block_average(
    df_list: List[pd.DataFrame],
    estimator: str = "MBAR",
    num: int = 10,
    **kwargs: Any,
) -> pd.DataFrame:
    """Free energy estimate for non-overlapping time blocks.

    Like :func:`alchemlyb.convergence.block_average`, but includes all ``num``
    blocks (including 90%-100%). Alchemlyb's implementation uses
    ``range(1, num)`` and omits the final block.

    Parameters
    ----------
    df_list : list of pandas.DataFrame
        List of ``u_nk`` or ``dHdl`` DataFrames, one per lambda window.
    estimator : {'MBAR', 'BAR', 'TI'}
        Free energy estimator name.
    num : int
        Number of equally sized time blocks per DataFrame.
    **kwargs
        Keyword arguments passed to the estimator constructor.

    Returns
    -------
    pandas.DataFrame
        Columns ``FE`` and ``FE_Error`` in reduced units (kT).
    """
    if estimator not in (FEP_ESTIMATORS + TI_ESTIMATORS):
        raise ValueError(
            f"Estimator {estimator} is not available in {FEP_ESTIMATORS + TI_ESTIMATORS}."
        )

    estimator_fit = _ESTIMATORS[estimator](**kwargs).fit
    logger.info("Using %s estimator for block averaging.", estimator)

    for i, df in enumerate(df_list):
        lambda_values = list({x[1:] for x in df.index.to_numpy()})
        if len(lambda_values) > 1:
            ind = next(
                j
                for j in range(len(lambda_values[0]))
                if len({x[j] for x in lambda_values}) > 1
            )
            raise ValueError(
                f"Provided DataFrame, df_list[{i}] has more than one lambda value "
                f"in df.index[{ind}]"
            )

    if estimator == "BAR" and len(df_list) > 2:
        raise ValueError(
            "Restrict to two DataFrames, one with a fep-lambda value and one its "
            "forward adjacent state for a meaningful result."
        )

    average_list: list[float] = []
    average_error_list: list[float] = []
    for i in range(1, num + 1):
        logger.info("Block average analysis: %.2f%%", 100 * i / num)
        sample = []
        for data in df_list:
            ind1 = len(data) // num * (i - 1)
            ind2 = len(data) if i == num else len(data) // num * i
            sample.append(data.iloc[ind1:ind2])
        sample_df = alchemlyb.concat(sample)
        result = estimator_fit(sample_df)

        average_list.append(result.delta_f_.iloc[0, -1])
        if estimator.lower() == "bar":
            average_error_list.append(np.nan)
        else:
            average_error_list.append(result.d_delta_f_.iloc[0, -1])

    convergence = pd.DataFrame(
        {"FE": average_list, "FE_Error": average_error_list},
    )
    convergence.attrs = df_list[0].attrs
    return convergence


@dataclass
class MBARResult:
    dg: float
    dg_std: float
    convergence: pd.DataFrame
    overlap: np.ndarray


def _extract_u_nk_out(out_file: str, temperature: float) -> pd.DataFrame:
    """Read reduced potentials from one Amber ``.out`` file."""
    try:
        u_nk = extract_u_nk(out_file, T=temperature)
        logger.info("Reading %d lines of u_nk from %s", len(u_nk), out_file)
        return u_nk
    except Exception as exc:
        msg = f"Error reading u_nk from {out_file}."
        logger.error(msg)
        raise OSError(msg) from exc


def _lambda_out_files(dirname: Path, prefix: str) -> list[str]:
    num_lambda = len(list(dirname.glob("lambda*")))
    return [
        str(dirname / f"lambda{i}/{prefix}/{prefix}.out")
        for i in range(num_lambda)
    ]


def annotate_convergence_to_conv_df(conv_df: pd.DataFrame):
    final_error = conv_df['Forward_Error'].iloc[-1]
    final_value = conv_df['Forward'].iloc[-1]
    convergence = np.logical_and(
        np.abs(conv_df['Forward'].values - final_value) <= final_error,
        np.abs(conv_df['Backward'].values - final_value) <= final_error
    )
    conv_df['Converged'] = convergence


def run_mbar(
    dirname: os.PathLike,
    prefix: str = 'prod',
    temperature: float = 298.15,
    decorrelate: bool = True,
    decorrelate_method: str = "dE",
    remove_burnin: bool = True,
    n_jobs: int = -1,
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
    prefix : str, optional
        Name of the production run subdirectory and output file prefix. For each lambda window,
        the function expects output files at `lambdaX/{prod_name}/{prod_name}.out`. Default is 'prod'.
    decorrelate : bool, optional
        If ``True``, subsample each ``u_nk`` with
        :func:`alchemlyb.preprocessing.decorrelate_u_nk` before analysis. Default is ``True``.
    decorrelate_method : {'all', 'dE'}, optional
        Energy series used for autocorrelation analysis. Default is ``'dE'``.
    remove_burnin : bool, optional
        If ``True``, discard unequilibrated frames before decorrelation. Default is ``True``.
    n_jobs : int, optional
        Number of parallel workers for reading ``.out`` files. ``-1`` uses all
        available workers (see :func:`joblib.Parallel`). Default is ``-1``.

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
    - Forward/backward convergence and block averaging are written to
      ``convergence.csv``. Plots are saved as ``convergence.png`` and
      ``block_average.png`` in the specified directory.
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
    out_files = _lambda_out_files(dirname, prefix)
    u_nks = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_extract_u_nk_out)(out_file, temperature)
        for out_file in out_files
    )
    with open(dirname / 'u_nks.pkl', 'wb') as f:
        pickle.dump(u_nks, f)

    if decorrelate:
        logger.info("Decorrelating u_nk data...")
        u_nks = decorrelate_u_nks(
            u_nks,
            method=decorrelate_method,
            remove_burnin=remove_burnin,
            n_jobs=n_jobs,
        )

    # evaluate free energy with MBAR
    logger.info("Running MBAR estimator...")
    mbarEstimator = MBAR()
    mbarEstimator.fit(alchemlyb.concat(u_nks))
    dg = mbarEstimator.delta_f_.iloc[0, -1] * kBT
    dg_std = mbarEstimator.d_delta_f_.iloc[0, -1] * kBT
    
    # convergence and block averaging analysis
    logger.info("Running convergence analysis...")
    conv_df = forward_backward_convergence(u_nks, "MBAR")
    for key in ['Forward', 'Forward_Error', 'Backward', 'Backward_Error']:
        conv_df[key] *= kBT

    logger.info("Running block averaging analysis...")
    block_df = block_average(u_nks, "MBAR")
    conv_df["Block_Average"] = block_df["FE"].to_numpy(dtype=float) * kBT
    conv_df["Block_Average_Error"] = block_df["FE_Error"].to_numpy(dtype=float) * kBT
    conv_df.attrs["energy_unit"] = "kcal/mol"
    annotate_convergence_to_conv_df(conv_df)
    conv_df.to_csv(dirname / "convergence.csv", index=None)
    save_convergence_plots(
        conv_df,
        dirname / "convergence.png",
        dirname / "block_average.png",
        title=f"Convergence Analysis - {dirname.name.capitalize()}",
        ylabel=r"$\Delta G$ (kcal/mol)",
        block_average_title=f"Block Average - {dirname.name.capitalize()}",
    )

    # overlap matrix
    logger.info("Plotting overlap matrix...")
    overlap_ax = plot_mbar_overlap_matrix(mbarEstimator.overlap_matrix)
    overlap_ax.figure.savefig(str(dirname /"overlap.png"), dpi=300)

    return MBARResult(dg, dg_std, conv_df, mbarEstimator.overlap_matrix)

