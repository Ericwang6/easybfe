import os
import logging
from typing import Optional
from pathlib import Path

from tqdm import tqdm
import alchemlyb
from alchemlyb.estimators import MBAR
from alchemlyb.parsing.amber import extract_u_nk
from alchemlyb.convergence import forward_backward_convergence
from alchemlyb.visualisation.convergence import plot_convergence
from alchemlyb.visualisation.mbar_matrix import plot_mbar_overlap_matrix

from ..cmd import init_logger


def run_mbar(
    dirname: os.PathLike,
    temperature: float = 298.15,
    logger: Optional[logging.Logger] = None,
    prefix: str = 'prod'
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
    logger = init_logger() if logger is None else logger
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
    conv_df = forward_backward_convergence(u_nks, "mbar")
    for key in ['Forward', 'Forward_Error', 'Backward', 'Backward_Error']:
        conv_df[key] *= kBT
    conv_df.to_csv(dirname / "convergence.csv", index=None)
    conv_ax = plot_convergence(conv_df)
    conv_ax.set_ylabel("$\Delta G$ (kcal/mol)")
    conv_ax.set_title(f"Convergence Analysis - {dirname.name.capitalize()}")
    conv_ax.figure.savefig(str(dirname /"convergence.png"), dpi=300)

    # overlap matrix
    logger.info("Plotting overlap matrix...")
    overlap_ax = plot_mbar_overlap_matrix(mbarEstimator.overlap_matrix)
    overlap_ax.figure.savefig(str(dirname /"overlap.png"), dpi=300)

    return dg, dg_std

    