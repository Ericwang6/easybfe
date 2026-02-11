import os
from pathlib import Path
import numpy as np
from .mbar import run_mbar
from alchemlyb.visualisation.convergence import plot_convergence


def analyze_abfe(directory: os.PathLike, prod_prefix: str, temperature: float = 298.15):
    wdir = Path(directory)
    results = {}
    for leg in ['complex', 'solvent', 'restraint']:
        results[leg] = run_mbar(wdir / leg, prod_prefix, temperature)
    boresch = float((wdir / 'boresch.dat').read_text().strip())

    dg = -results['complex'].dg + results['solvent'].dg + results['restraint'].dg + boresch
    dg_std = np.linalg.norm([results['complex'].dg_std, results['solvent'].dg_std, results['restraint'].dg_std])

    conv_df = results['complex'].convergence.copy()
    for fw in ['Forward', 'Backward']:
        conv_df[fw] = -results['complex'].convergence[fw] + results['solvent'].convergence[fw] + results['restraint'].convergence[fw] + boresch
        fw_err = fw + '_Error'
        conv_df[fw_err] = np.sqrt(results['complex'].convergence[fw_err].values ** 2 + \
            results['solvent'].convergence[fw_err].values ** 2 + \
            results['restraint'].convergence[fw_err].values ** 2
        )
    
    conv_df.to_csv(wdir / "convergence.csv", index=None)
    conv_ax = plot_convergence(conv_df)
    conv_ax.set_ylabel("$\Delta G$ (kcal/mol)")
    conv_ax.set_title(f"ABFE Convergence Analysis - {wdir.name.capitalize()}")
    conv_ax.figure.savefig(str(wdir /"convergence.png"), dpi=300)
    
    (wdir / 'result.dat').write_text(f'{dg:.4f}\n{dg_std:.4f}')
    
