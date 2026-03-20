import os
import json
from pathlib import Path
import numpy as np
from .mbar import run_mbar, plot_convergence


def analyze_rbfe(directory: os.PathLike, prod_prefix: str = '05.prod', temperature: float = 298.15, force_run: bool = True):
    wdir = Path(directory)

    if not force_run and (wdir / 'result.json').is_file():
        with (wdir / 'result.json').open('r') as f:
            return json.load(f)

    results = {}
    json_data  ={}
    for leg in ['complex', 'solvent', 'gas']:
        if not (wdir / leg / 'done.tag').is_file():
            continue
        results[leg] = run_mbar(wdir / leg, prod_prefix, temperature)
        json_data['dg_'+leg] = results[leg].dg
        json_data['dg_'+leg+'_std'] = results[leg].dg_std

    leg_pairs = [
        ('complex', 'solvent', 'total'), 
        ('complex', 'gas', 'complex'), 
        ('solvent', 'gas', 'solvation')
    ]
    for leg1, leg2, name in leg_pairs:
        if not (leg1 in results and leg2 in results):
            continue

        ddg = results[leg1].dg - results[leg2].dg
        ddg_std = np.linalg.norm([results[leg1].dg_std, results[leg2].dg_std])

        json_data[f'ddg_{name}'] = ddg
        json_data[f'ddg_{name}_std'] = ddg_std

        conv_df = results[leg1].convergence.copy()
        for fw in ['Forward', 'Backward']:
            conv_df[fw] = results[leg1].convergence[fw] - results[leg2].convergence[fw]
            fw_err = fw + '_Error'
            conv_df[fw_err] = np.sqrt(results[leg1].convergence[fw_err].values ** 2 + results[leg2].convergence[fw_err].values ** 2)

        conv_df.to_csv(wdir / f"{name}_convergence.csv", index=None)
        conv_ax = plot_convergence(conv_df)
        conv_ax.set_ylabel(r"$\Delta\Delta G$ (kcal/mol)")
        suffix = f' ({name.capitalize()}) ' if name != 'total' else ''
        conv_ax.set_title(
            f"RBFE Convergence: {wdir.name}{suffix}",
            fontsize=14,
            fontweight="semibold",
            pad=12,
        )
        conv_ax.figure.tight_layout(rect=(0, 0, 1, 0.97))
        conv_ax.figure.savefig(str(wdir / f"{name}_convergence.png"), dpi=300)

    with (wdir / "result.json").open("w") as f:
        json.dump(json_data, f, indent=4)

    return json_data

    
