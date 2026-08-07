import os
import json
import logging
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .mbar import run_mbar, save_convergence_plots, annotate_convergence_to_conv_df, MBARResult
from .remd import parse_remlog
from .trajectory import post_process_trajectory, compute_rmsd, plot_rmsd
from .interaction import analyze_interactions_for_trajectory, plot_interactions
from .boresch import analyze_boresch_lambda, lambda_directories, plot_boresch_coordinates


logger = logging.getLogger(__name__)

LEGS = ("complex", "solvent", "restraint")


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
    """Topology used to read a leg's trajectory.

    The built-system PDB is preferred over the Amber prmtop because it carries the
    chain identifiers and residue numbers inherited from the input protein (a
    prmtop has none, so MDAnalysis would renumber residues from 1). This mirrors
    the plain-MD analysis path, which also reads ``<basename>.pdb``.
    """
    for preferred in (leg_directory / "system.pdb", leg_directory / "system.prmtop"):
        if preferred.is_file():
            return preferred
    for pattern in ("*.pdb", "*.prmtop"):
        topologies = sorted(leg_directory.glob(pattern))
        if topologies:
            return max(topologies, key=lambda path: path.stat().st_size)
    raise FileNotFoundError(f"No topology found in {leg_directory}")


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


def _floats(values) -> list[float]:
    """JSON-friendly float list (``NaN``/``inf`` become ``None``)."""
    return [
        None if not np.isfinite(value) else float(value)
        for value in np.asarray(values, dtype=float)
    ]


def _convergence_summary(conv_df: pd.DataFrame) -> dict:
    """Forward/backward convergence of one free energy series as plain data.

    ``is_converged`` reuses the criterion of
    :func:`~easybfe.analysis.mbar.annotate_convergence_to_conv_df` (forward and
    backward estimates within the final forward error of the final estimate) and
    requires it to hold over the whole second half of the simulation, not just
    at the last point.
    """
    fractions = conv_df["data_fraction"].to_numpy(dtype=float)
    converged = conv_df["Converged"].to_numpy(dtype=bool)
    second_half = fractions >= 0.5
    return {
        "data_fraction": _floats(fractions),
        "forward": _floats(conv_df["Forward"]),
        "forward_error": _floats(conv_df["Forward_Error"]),
        "backward": _floats(conv_df["Backward"]),
        "backward_error": _floats(conv_df["Backward_Error"]),
        "converged": [bool(value) for value in converged],
        "is_converged": bool(converged[second_half].all()) if second_half.any() else False,
        "final_forward": float(conv_df["Forward"].iloc[-1]),
        "final_forward_error": float(conv_df["Forward_Error"].iloc[-1]),
        "final_backward": float(conv_df["Backward"].iloc[-1]),
        "final_backward_error": float(conv_df["Backward_Error"].iloc[-1]),
    }


def _block_average_summary(conv_df: pd.DataFrame) -> dict:
    """Block-averaged free energies of one series as plain data.

    ``std`` is the spread of the per-block estimates (the band drawn by
    :func:`~easybfe.analysis.mbar.plot_block_average`), which is an
    estimator-independent measure of how much the estimate drifts across the
    trajectory.
    """
    blocks = conv_df["Block_Average"].to_numpy(dtype=float)
    finite = blocks[np.isfinite(blocks)]
    return {
        "data_fraction": _floats(conv_df["data_fraction"]),
        "values": _floats(blocks),
        "errors": _floats(conv_df["Block_Average_Error"]),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
    }


def _overlap_summary(overlap: np.ndarray) -> dict:
    """Nearest-neighbour overlap of an MBAR overlap matrix.

    Only the sub-/super-diagonal entries matter in practice: a small
    ``min_adjacent`` flags a lambda gap where the windows barely share phase
    space, which is the usual cause of a poorly determined leg. Judge the values
    against ``1 / n_states``, the overlap of a ladder whose windows sample the
    very same distribution (0.042 for 24 windows), not against 1.
    """
    matrix = np.asarray(overlap, dtype=float)
    n_states = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if n_states < 2:
        return {
            "n_states": n_states,
            "adjacent": [],
            "min_adjacent": None,
            "mean_adjacent": None,
        }
    adjacent = np.diag(matrix, k=1)
    return {
        "n_states": n_states,
        "adjacent": _floats(adjacent),
        "min_adjacent": float(np.min(adjacent)),
        "mean_adjacent": float(np.mean(adjacent)),
    }


def _leg_summary(result: MBARResult, leg_directory: Path, prod_prefix: str) -> dict:
    """Per-leg diagnostics: free energy, convergence, overlap, exchange rate."""
    summary = {
        "dg": float(result.dg),
        "dg_std": float(result.dg_std),
        "convergence": _convergence_summary(result.convergence),
        "block_average": _block_average_summary(result.convergence),
        "overlap": _overlap_summary(result.overlap),
    }
    # The remlog only exists for H-REMD stages (``use_remd: true``); a plain
    # (e.g. pre-production) stage has no exchanges to report.
    summary["exchange"] = parse_remlog(leg_directory / f"{prod_prefix}.log")
    return summary


def _leg_status(leg_directory: Path) -> dict:
    """Read a leg's ``status.json``, the file its ``run.sh`` maintains."""
    try:
        return json.loads((leg_directory / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def _format_leg_status(status: dict) -> str:
    """One-line summary of a leg's status for the log."""
    summary = f"{status.get('state', 'unknown')} at stage {status.get('stage', '?')}"
    excerpt = status.get("error_excerpt")
    if excerpt:
        summary += f" -- {excerpt.splitlines()[0]}"
    return summary


def _infer_early_stop(wdir: Path) -> bool:
    """True when the legs stopped after the pre-production phase.

    A leg's ``run.sh`` writes ``preprod.done.tag`` after the second-to-last
    stage and ``done.tag`` after the last one, so a leg carrying the former but
    not the latter never ran its production stage.
    """
    leg_dirs = [wdir / leg for leg in LEGS if (wdir / leg).is_dir()]
    if not leg_dirs:
        return False
    return all(
        (leg_dir / "preprod.done.tag").is_file() and not (leg_dir / "done.tag").is_file()
        for leg_dir in leg_dirs
    )


def analyze_abfe(
    directory: os.PathLike,
    prod_prefix: str = "05.prod",
    temperature: float = 298.15,
    force_run: bool = False,
    n_jobs: int = -1,
    done_tag: str = "done.tag",
    run_trajectory_analysis: bool = True,
    early_stop: Optional[bool] = None,
):
    """Analyze a finished ABFE directory and write ``result.json``.

    Parameters
    ----------
    directory : os.PathLike
        ABFE directory holding the ``complex``/``solvent``/``restraint`` legs,
        ``boresch.dat`` and (on output) ``result.json``.
    prod_prefix : str, optional
        Production stage name to analyze (also the remlog basename).
    temperature : float, optional
        Simulation temperature in Kelvin.
    force_run : bool, optional
        Recompute even when ``result.json`` already exists.
    n_jobs : int, optional
        Workers for the trajectory analyses. ``-1`` uses up to 4.
    done_tag : str, optional
        Per-leg completion tag gating which legs are analyzed.
    run_trajectory_analysis : bool, optional
        Run the endpoint trajectory / interaction / Boresch analyses.
    early_stop : bool, optional
        Value of the ``early_stop`` field in ``result.json``. Inferred from the
        per-leg completion tags when ``None`` (see :func:`_infer_early_stop`).

    Returns
    -------
    dict
        Per-leg and total free energies, plus the ``early_stop`` flag and
        ``convergence``/``block_average``/``legs`` diagnostics. Empty when a leg
        is missing.
    """
    wdir = Path(directory)

    if not force_run and (wdir / 'result.json').is_file():
        with (wdir / 'result.json').open('r') as f:
            res = json.load(f)
        if run_trajectory_analysis:
            _run_trajectory_analysis(wdir, prod_prefix, n_jobs, force_run=False)
        return res

    results = {}
    missing = []
    for leg in LEGS:
        if not (wdir / leg / done_tag).is_file():
            missing.append(leg)
            continue
        results[leg] = run_mbar(wdir / leg, prod_prefix, temperature)

    if missing:
        # Say which leg is missing and why: a silent empty result here used to
        # send people looking for a bug in the analysis instead of at the leg
        # that never finished.
        logger.error(
            "Cannot compute the total dG: leg(s) %s have no %s. Per-leg state:",
            ", ".join(missing), done_tag,
        )
        for leg in missing:
            status = _leg_status(wdir / leg)
            logger.error(
                "  %-9s %s", leg,
                _format_leg_status(status) if status else f"no status.json in {wdir / leg}",
            )
        return {}

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
        "early_stop": _infer_early_stop(wdir) if early_stop is None else bool(early_stop),
        "prod_prefix": prod_prefix,
        "temperature": temperature,
        # Diagnostics of the combined dG (the per-leg series propagated into
        # conv_df above) followed by the per-leg ones.
        "convergence": _convergence_summary(conv_df),
        "block_average": _block_average_summary(conv_df),
        "legs": {
            leg: _leg_summary(results[leg], wdir / leg, prod_prefix)
            for leg in LEGS
        },
    }

    with (wdir / "result.json").open("w") as f:
        json.dump(res, f, indent=4)

    if run_trajectory_analysis:
        _run_trajectory_analysis(wdir, prod_prefix, n_jobs, force_run=force_run)
    return res
    
