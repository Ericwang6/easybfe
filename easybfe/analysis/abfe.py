import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .mbar import run_mbar, save_convergence_plots, annotate_convergence_to_conv_df
from .trajectory import post_process_trajectory, compute_rmsd, plot_rmsd
from .interaction import analyze_interactions_for_trajectory, plot_interactions
from .boresch import analyze_boresch_lambda, lambda_directories, plot_boresch_coordinates


def _cleanup_mdanalysis_offsets(wdir: Path, prod_prefix: str) -> None:
    for prod_directory in wdir.glob(f"*/lambda*/{prod_prefix}"):
        for pattern in (".prod*.npz", ".prod*.lock"):
            for path in prod_directory.glob(pattern):
                path.unlink(missing_ok=True)


def _worker_count(n_jobs: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    if n_jobs < 0:
        return min(4, task_count)
    return min(n_jobs, task_count)


def _topology_path(leg_directory: Path) -> Path:
    preferred = leg_directory / "system.prmtop"
    if preferred.is_file():
        return preferred
    topologies = sorted(leg_directory.glob("*.prmtop"))
    if not topologies:
        raise FileNotFoundError(f"No Amber topology found in {leg_directory}")
    return max(topologies, key=lambda path: path.stat().st_size)


def _endpoint_trajectory_analysis(
    leg: str,
    leg_directory: Path,
    lambda_directory: Path,
    prod_prefix: str,
    force_run: bool,
) -> tuple[str, Path]:
    prod_directory = lambda_directory / prod_prefix
    trajectory = prod_directory / f"{prod_prefix}.mdcrd"
    if not trajectory.is_file():
        return leg, prod_directory

    processed_pdb = prod_directory / "prod_processed.pdb"
    processed_xtc = prod_directory / "prod_processed.xtc"
    has_protein = leg in ("complex", "restraint")
    if force_run or not processed_pdb.is_file() or not processed_xtc.is_file():
        post_process_trajectory(
            in_top=str(_topology_path(leg_directory)),
            in_trj=str(trajectory),
            out_pdb=str(processed_pdb),
            out_trj=str(processed_xtc),
            process_pbc=True,
            do_alignment=has_protein,
            in_trj_format="NCDF",
            center_selection="protein" if has_protein else "resname MOL",
            output_selection="protein or resname MOL" if has_protein else "resname MOL",
            align_selection="backbone",
            include_water_selection="resname MOL"
        )
    rmsd_data = compute_rmsd(
        top=str(processed_pdb),
        trj=str(processed_xtc),
        selection="resindex 0",
        use_symmetry_correction=False,
        save_path=str(prod_directory / "prod_rmsd.txt"),
    )
    ax = plot_rmsd(
        rmsd_data,
        name=f"{leg.capitalize()} {lambda_directory.name}",
        save_path=str(prod_directory / "prod_rmsd.png"),
        dpi=300,
    )
    plt.close(ax.figure)
    return leg, prod_directory


def _run_trajectory_analysis(wdir: Path, prod_prefix: str, n_jobs: int, force_run: bool) -> None:
    try:
        endpoint_tasks = []
        lambda_dirs_by_leg: dict[str, list[Path]] = {}
        for leg in ("complex", "restraint", "solvent"):
            leg_directory = wdir / leg
            lambda_dirs = lambda_directories(leg_directory)
            lambda_dirs_by_leg[leg] = lambda_dirs
            if not lambda_dirs:
                continue
            for lambda_directory in (lambda_dirs[0], lambda_dirs[-1]):
                endpoint_tasks.append((leg, leg_directory, lambda_directory))

        Parallel(n_jobs=_worker_count(n_jobs, len(endpoint_tasks)))(
            delayed(_endpoint_trajectory_analysis)(
                leg, leg_directory, lambda_directory, prod_prefix, force_run
            )
            for leg, leg_directory, lambda_directory in endpoint_tasks
        )

        restraint_lambdas = lambda_dirs_by_leg.get("restraint", [])
        if restraint_lambdas:
            prod_directory = restraint_lambdas[-1] / prod_prefix
            processed_pdb = prod_directory / "prod_processed.pdb"
            processed_xtc = prod_directory / "prod_processed.xtc"
            interaction_csv = prod_directory / "interaction.csv"
            interaction_png = prod_directory / "interaction.png"
            if (
                processed_pdb.is_file()
                and processed_xtc.is_file()
                and (force_run or not interaction_csv.is_file() or not interaction_png.is_file())
            ):
                interaction_df = analyze_interactions_for_trajectory(
                    top=str(processed_pdb),
                    trj=str(processed_xtc),
                    out_csv=str(interaction_csv),
                    use_mpi=True,
                    remove_tmp=True,
                )
                if not interaction_df.empty:
                    ax = plot_interactions(
                        interaction_df,
                        title=f"Restraint {restraint_lambdas[-1].name} interactions",
                        save_path=str(interaction_png),
                        dpi=300,
                    )
                    plt.close(ax.figure)

        boresch_outputs = [
            path
            for leg in ("complex", "restraint")
            for path in (wdir / leg / "boresch.csv", wdir / leg / "boresch.png")
        ]
        if not force_run and all(path.is_file() for path in boresch_outputs):
            return

        boresch_tasks = []
        for leg in ("complex", "restraint"):
            leg_directory = wdir / leg
            if not leg_directory.is_dir():
                continue
            topology = _topology_path(leg_directory)
            for lambda_directory in lambda_dirs_by_leg.get(leg, []):
                boresch_tasks.append((leg, topology, lambda_directory))

        frames = Parallel(n_jobs=_worker_count(n_jobs, len(boresch_tasks)))(
            delayed(analyze_boresch_lambda)(leg, topology, lambda_directory, prod_prefix)
            for leg, topology, lambda_directory in boresch_tasks
        )
        if frames:
            boresch_df = pd.concat(frames, ignore_index=True)
            for leg in ("complex", "restraint"):
                leg_df = boresch_df[boresch_df["leg"] == leg].reset_index(drop=True)
                leg_directory = wdir / leg
                leg_df.to_csv(leg_directory / "boresch.csv", index=False)
                plot_boresch_coordinates(leg_df, leg, leg_directory / "boresch.png")
    finally:
        _cleanup_mdanalysis_offsets(wdir, prod_prefix)


def analyze_abfe(
    directory: os.PathLike,
    prod_prefix: str = "05.prod",
    temperature: float = 298.15,
    force_run: bool = False,
    n_jobs: int = -1,
):
    wdir = Path(directory)

    if not force_run and (wdir / 'result.json').is_file():
        with (wdir / 'result.json').open('r') as f:
            res = json.load(f)
        _run_trajectory_analysis(wdir, prod_prefix, n_jobs, force_run=False)
        return res

    results = {}
    for leg in ['complex', 'solvent', 'restraint']:
        if not (wdir / leg / 'done.tag').is_file():
            continue
        results[leg] = run_mbar(wdir / leg, prod_prefix, temperature)
    boresch = float((wdir / 'boresch.dat').read_text().strip())

    if 'complex' not in results or 'solvent' not in results or 'restraint' not in results:
        return {}

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

    conv_df["Block_Average"] = (
        -results['complex'].convergence["Block_Average"]
        + results['solvent'].convergence["Block_Average"]
        + results['restraint'].convergence["Block_Average"]
        + boresch
    )
    conv_df["Block_Average_Error"] = np.sqrt(
        results['complex'].convergence["Block_Average_Error"].values ** 2
        + results['solvent'].convergence["Block_Average_Error"].values ** 2
        + results['restraint'].convergence["Block_Average_Error"].values ** 2
    )

    conv_df.to_csv(wdir / "convergence.csv", index=None)
    annotate_convergence_to_conv_df(conv_df)
    save_convergence_plots(
        conv_df,
        wdir / "convergence.png",
        wdir / "block_average.png",
        title=f"ABFE Convergence: {wdir.name.capitalize()}",
        ylabel=r"$\Delta G$ (kcal/mol)",
        block_average_title=f"ABFE Block Average: {wdir.name.capitalize()}",
    )

    res = {
        "complex": results["complex"].dg,
        "complex_std": results["complex"].dg_std,
        "solvent": results["solvent"].dg,
        "solvent_std": results["solvent"].dg_std,
        "restraint": results["restraint"].dg,
        "restraint_std": results["restraint"].dg_std,
        "boresch": boresch,
        "total": dg,
        "total_std": dg_std,
    }

    with (wdir / "result.json").open("w") as f:
        json.dump(res, f, indent=4)

    _run_trajectory_analysis(wdir, prod_prefix, n_jobs, force_run=force_run)
    return res
    
