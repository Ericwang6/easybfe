import os
import json
from pathlib import Path
from typing import Optional, Any
import logging
import matplotlib.pyplot as plt
import warnings

import MDAnalysis as mda
warnings.filterwarnings("ignore", module="MDAnalysis.coordinates.PDB")
warnings.filterwarnings("ignore", module="MDAnalysis.coordinates.XDR")
import numpy as np

from .trajectory import post_process_trajectory, compute_rmsd, plot_rmsd
from .interaction import analyze_interactions_for_trajectory, plot_interactions
from ..config import AnalysisConfig
from ..core.ligand import Ligand
from ..core.protein import Protein
from ..gbsa import GBSARunner


logger = logging.getLogger(__name__)


def _load_positions(topology: Path, trajectory: Path) -> np.ndarray:
    """
    Load all frames from a trajectory into a NumPy array.

    Parameters
    ----------
    topology : Path
        Path to topology file (e.g., processed PDB).
    trajectory : Path
        Path to trajectory file (e.g., processed XTC).

    Returns
    -------
    numpy.ndarray
        Array of positions with shape (n_frames, n_atoms, 3) in Angstrom.
    """
    universe = mda.Universe(str(topology), str(trajectory))
    frames: list[np.ndarray] = []
    for ts in universe.trajectory:
        frames.append(ts.positions.copy())
    if not frames:
        raise ValueError(f"No frames found in trajectory: {trajectory}")
    return np.asarray(frames, dtype=np.float64)


def run_plain_md_analysis_workflow(
    directory: os.PathLike,
    config: Optional[AnalysisConfig] = None,
    prefix: Optional[str] = None,
):
    """
    Run MD analysis workflow on a trajectory.

    Parameters
    ----------
    directory : os.PathLike
        Path to the MD task directory containing config.json and trajectory files
    prefix : str, optional
        Production stage name (e.g., '05.prod'). If not provided, will be read from config.json.
        Default is '05.prod'.
    config : AnalysisConfig, optional
        Analysis configuration. If None, defaults from config.json will be used.

    Returns
    -------
    None
    """
    directory = Path(directory)
    
    # Read MD config
    md_cfg_path = directory / 'config.json'
    if not md_cfg_path.is_file():
        raise RuntimeError(f"config.json not found in {directory}")
    
    with open(md_cfg_path) as f:
        md_config: dict[str, Any] = json.load(f)
    
    # Get production stage name
    if 'workflow' in md_config and len(md_config['workflow']) > 0:
        prod_stage = md_config['workflow'][-1]['name']
    elif prefix:
        prod_stage = prefix
        logger.warning(f"Could not determine production stage from config.json, using prefix: {prefix}")
    else:
        raise RuntimeError(f"Cannot infer the prod stage from {directory}. Please provide the `prefix` argument")
    
    # Assemble real analysis config
    task_type = md_config.get("task_type", None)
    ana_config = md_config.get("analysis", dict())
    if config:
        ana_config.update(config.model_dump())
    if ana_config.get('task_type', None) is None:
        ana_config['task_type'] = task_type
    if ana_config.get('task_type', None) is not None and ana_config['task_type'] != task_type:
        logger.warning(f"Analysis task_type {ana_config['task_type']} is not same to MD task_type ({task_type})")
    
    # set names
    task_name = md_config.get("task_name", directory.name)
    ana_config = AnalysisConfig.model_validate(ana_config)
    if ana_config.rmsd_name is None:
        ana_config.rmsd_name = task_name
    if ana_config.interaction_name is None:
        ana_config.interaction_name = task_name
    
    # Process PBC and do alignment
    post_process_trajectory(
        in_top=str(directory / 'system.pdb'),
        in_trj=str(directory / prod_stage / f'{prod_stage}.mdcrd'),
        out_pdb=str(directory / prod_stage / 'prod_processed.pdb'),
        out_trj=str(directory / prod_stage / 'prod_processed.xtc'),
        ref=None,
        process_pbc=ana_config.process_pbc,
        do_alignment=ana_config.do_alignment,
        in_top_format=None,
        in_trj_format='NCDF',
        ref_format=None,
        center_selection=ana_config.center_selection,
        output_selection=ana_config.output_selection,
        align_selection=ana_config.align_selection,
        remove_tmp=ana_config.remove_tmp
    )
    
    # Compute & plot RMSD
    rmsd_data = compute_rmsd(
        top=str(directory / prod_stage / 'prod_processed.pdb'),
        trj=str(directory / prod_stage / 'prod_processed.xtc'),
        ref='',
        selection=ana_config.rmsd_selection,
        use_symmetry_correction=ana_config.use_symmetry_correction,
        heavy_atoms_only=ana_config.heavy_atoms_only,
        save_path=str(directory / prod_stage / 'prod_rmsd.txt')
    )
    
    ax = plot_rmsd(
        data=rmsd_data,
        name=ana_config.rmsd_name,
        save_path=str(directory / prod_stage / 'prod_rmsd.png'),
        dpi=450
    )
    plt.close(ax.figure)
    
    # Interactions
    if ana_config.interaction_analysis:
        interact_df = analyze_interactions_for_trajectory(
            top=str(directory / prod_stage / 'prod_processed.pdb'),
            trj=str(directory / prod_stage / 'prod_processed.xtc'),
            out_csv=str(directory / prod_stage / 'interaction.csv'),
            use_mpi=ana_config.use_mpi,
            remove_tmp=ana_config.remove_tmp,
            use_strict_hbond=ana_config.use_strict_hbond,
            resnr_renum=ana_config.resnr_renum
        )
        
        ax = plot_interactions(
            interact_df,
            title=ana_config.interaction_name,
            save_path=str(directory / prod_stage / 'interaction.png'),
            dpi=450
        )
        plt.close(ax.figure)

    # GBSA binding free energy calculation
    if ana_config.do_gbsa:
        protein_pdb_path = Path(directory / 'protein.pdb')
        if not protein_pdb_path.is_file():
            logger.warning(
                "GBSA protein PDB file not found at %s; skipping GBSA calculation.",
                protein_pdb_path,
            )
            return

        try:
            ligand = Ligand.from_directory(directory / 'ligand')
            protein = Protein.from_pdb(protein_pdb_path)

            protein_ffs = md_config.get("protein_ff", ["amber14-all.xml", "amber14/tip3p.xml"])
            saltcon = md_config.get('ionic_strength', 0.15)
            if saltcon != ana_config.gbsa_saltcon:
                logger.warning(f"GBSA use a different salt concentration ({ana_config.gbsa_saltcon}) but MD use ({saltcon})")

            runner = GBSARunner(
                protein=protein,
                ligand=ligand,
                protein_ffs=protein_ffs,
                igb=ana_config.gbsa_igb,
                saltcon=ana_config.gbsa_saltcon,
                epsin=ana_config.gbsa_epsin,
                epsout=ana_config.gbsa_epsout,
                temperature=ana_config.gbsa_temperature,
            )

            top_path = directory / prod_stage / "prod_processed.pdb"
            trj_path = directory / prod_stage / "prod_processed.xtc"

            if not top_path.is_file() or not trj_path.is_file():
                logger.warning(
                    "Processed topology or trajectory not found (%s, %s); skipping GBSA calculation.",
                    top_path,
                    trj_path,
                )
                return

            all_positions = _load_positions(topology=top_path, trajectory=trj_path)

            n_ligand_atoms = ligand.to_openmm().topology.getNumAtoms()
            if n_ligand_atoms <= 0:
                logger.warning("Ligand atom count is non-positive (%d); skipping GBSA.", n_ligand_atoms)
                return

            if all_positions.shape[1] <= n_ligand_atoms:
                logger.warning(
                    "Total atoms (%d) <= ligand atoms (%d); cannot split protein/ligand for GBSA.",
                    all_positions.shape[1],
                    n_ligand_atoms,
                )
                return

            ligand_positions = all_positions[:, :n_ligand_atoms, :]
            protein_positions = all_positions[:, n_ligand_atoms:, :]

            energies = runner.compute_multiple_frames(
                protein_positions=protein_positions,
                ligand_positions=ligand_positions,
                progress_bar=True,
            )

            out_path = directory / prod_stage / "gbsa.txt"
            np.savetxt(out_path, energies, fmt="%.10f")
            logger.info("GBSA calculation finished; wrote %d frames to %s", len(energies), out_path)
        except Exception:
            logger.exception("GBSA calculation failed for directory %s", directory)