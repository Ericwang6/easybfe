import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .mbar import run_mbar, save_convergence_plots, annotate_convergence_to_conv_df
from .mle import maximum_likelihood_estimator

logger = logging.getLogger(__name__)


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

        conv_df["Block_Average"] = (
            results[leg1].convergence["Block_Average"]
            - results[leg2].convergence["Block_Average"]
        )
        conv_df["Block_Average_Error"] = np.sqrt(
            results[leg1].convergence["Block_Average_Error"].values ** 2
            + results[leg2].convergence["Block_Average_Error"].values ** 2
        )

        conv_df.to_csv(wdir / f"{name}_convergence.csv", index=None)
        suffix = f' ({name.capitalize()}) ' if name != 'total' else ''
        annotate_convergence_to_conv_df(conv_df)
        save_convergence_plots(
            conv_df,
            wdir / f"{name}_convergence.png",
            wdir / f"{name}_block_average.png",
            title=f"RBFE Convergence: {wdir.name}{suffix}",
            ylabel=r"$\Delta\Delta G$ (kcal/mol)",
            block_average_title=f"RBFE Block Average: {wdir.name}{suffix}",
        )

    with (wdir / "result.json").open("w") as f:
        json.dump(json_data, f, indent=4)

    return json_data


def analyze_rbfe_dg_network(
    directory: os.PathLike,
    *,
    bias: float = 0.0,
    result_filename: str = "result.json",
    dg_csv_name: str = "dg.csv",
) -> pd.DataFrame:
    """Aggregate pairwise RBFE results under ``directory`` and estimate per-ligand dG via MLE.

    Each immediate subdirectory whose name matches ``*~*`` is treated as one edge
    (ligand A ``~`` ligand B). Directories without ``result.json`` are skipped.
    Expected keys when present (from :func:`analyze_rbfe`): ``ddg_total``,
    ``ddg_total_std``.

    Parameters
    ----------
    directory : os.PathLike
        Parent directory containing edge subdirectories (e.g. batch output base).
    bias : float, optional
        Added to all estimated dG values (passed to :func:`maximum_likelihood_estimator`).
    result_filename : str, optional
        JSON file name inside each edge directory. Default: ``result.json``.
    dg_csv_name : str, optional
        Written under ``directory``. Default: ``dg.csv``.

    Returns
    -------
    pandas.DataFrame
        Columns ``ligand``, ``dG``, ``dG_std`` (same as MLE output).

    Raises
    ------
    ValueError
        If no ``*~*`` subdirectories exist under ``directory``, or none of them contain
        a readable ``result.json`` with the required keys.
    """
    root = Path(directory)
    edge_dirs = sorted(p for p in root.glob("*~*") if p.is_dir())
    if not edge_dirs:
        raise ValueError(
            f"No subdirectories matching *~* under {root!s} (expected one folder per edge)."
        )

    rows: list[dict[str, object]] = []
    for sub in edge_dirs:
        name = sub.name
        if "~" not in name:
            continue
        ligand_a, ligand_b = name.split("~", 1)
        result_path = sub / result_filename
        if not result_path.is_file():
            logger.debug("Skipping %s: missing %s", sub, result_filename)
            continue
        with result_path.open("r") as f:
            data = json.load(f)
        try:
            ddg = float(data["ddg_total"])
            ddg_std = float(data["ddg_total_std"])
        except KeyError as e:
            raise ValueError(
                f"{result_path} must contain ddg_total and ddg_total_std (from analyze_rbfe)."
            ) from e
        rows.append(
            {
                "ligandA_name": ligand_a,
                "ligandB_name": ligand_b,
                "ddG.total": ddg,
                "ddG_std.total": ddg_std,
            }
        )

    n_scanned = len(edge_dirs)
    n_with_result = len(rows)
    unique_ligands = set()
    for row in rows:
        unique_ligands.add(row["ligandA_name"])
        unique_ligands.add(row["ligandB_name"])
    logger.info(
        "RBFE dg-network under %s: %d perturbation(s) with %s out of %d *~* director(ies); "
        "%d unique ligand(s).",
        root,
        n_with_result,
        result_filename,
        n_scanned,
        len(unique_ligands),
    )

    if not rows:
        raise ValueError(
            f"No {result_filename} found under any *~* subdirectory of {root!s} "
            "(nothing to aggregate)."
        )

    df_edges = pd.DataFrame(rows)
    dg_df = maximum_likelihood_estimator(df_edges, bias=bias)
    dg_df.to_csv(root / dg_csv_name, index=False)
    return dg_df

