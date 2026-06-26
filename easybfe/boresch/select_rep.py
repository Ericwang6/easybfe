"""Select a representative frame from a protein-ligand complex trajectory.

The representative frame is the trajectory frame whose ligand rotatable-torsion
geometry is closest to a per-torsion reference conformation. The reference value
of each torsion is its median (an actually observed sample, see
:func:`easybfe.boresch.utils.median_reference`), so no circular averaging is
needed to build the reference. Closeness is then measured with a circular
distance against that reference, so the periodic nature of dihedral angles is
handled correctly.

The numerical helpers used here (rotatable-torsion detection, the median
reference and the torsion-distribution plot) live in
:mod:`easybfe.boresch.utils`.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional

import numpy as np

from .utils import (
    median_reference,
    find_rotatable_torsions,
    compute_torsions_along_trajectory,
    compute_frame_deviations,
    plot_torsion_distributions,
)


logger = logging.getLogger(__name__)


def _format_selection_info(
    rep_frame: int,
    deviations: np.ndarray,
    reference: np.ndarray,
    rep_values: np.ndarray,
    torsion_atoms,
) -> str:
    """Build a human-readable report of the representative-frame selection.

    The report lists, for every rotatable torsion, its four atom indices, the
    per-torsion median reference and the value at the selected representative
    frame (all in degrees), followed by the deviation of the selected frame.

    Parameters
    ----------
    rep_frame : int
        Index of the selected representative frame.
    deviations : numpy.ndarray
        Per-frame circular deviation, shape ``(n_frames,)``.
    reference : numpy.ndarray
        Per-torsion median reference in radians, shape ``(M,)``.
    rep_values : numpy.ndarray
        Torsion values at the representative frame in radians, shape ``(M,)``.
    torsion_atoms : sequence of tuple of int
        List of torsion atom-index 4-tuples.

    Returns
    -------
    str
        The assembled multi-line report.
    """
    reference_deg = np.rad2deg(np.asarray(reference, dtype=float))
    rep_deg = np.rad2deg(np.asarray(rep_values, dtype=float))

    lines = [
        "Representative-frame selection (ligand rotatable torsions)",
        f"  n_frames           = {deviations.shape[0]}",
        f"  n_torsions         = {len(torsion_atoms)}",
        f"  representative frame = {rep_frame}",
        f"  frame deviation    = {float(deviations[rep_frame]):.4f} "
        f"(min {float(np.min(deviations)):.4f}, max {float(np.max(deviations)):.4f})",
        "",
        f"  {'torsion (atoms)':<24}{'median (deg)':>14}{'rep (deg)':>14}",
        f"  {'-' * 24}{'-' * 14:>14}{'-' * 14:>14}",
    ]
    for atoms, ref_deg, value_deg in zip(torsion_atoms, reference_deg, rep_deg):
        label = "-".join(str(i) for i in atoms)
        lines.append(f"  {label:<24}{ref_deg:>14.2f}{value_deg:>14.2f}")
    return "\n".join(lines)


def select_representative_frame(
    topology: os.PathLike,
    trajectory: os.PathLike,
    ligand_sdf: Optional[os.PathLike] = None,
    ligand_mol: Optional["object"] = None,
    ligand_selection: Optional[str] = None,
    protein_selection: str = "protein",
    out_pdb: os.PathLike = "",
    out_sdf: os.PathLike = "",
    out_fig: os.PathLike = "",
    out_info: os.PathLike = "",
    topology_format: Optional[str] = None,
    trajectory_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Select the representative frame of a complex trajectory by ligand torsions.

    The ligand rotatable torsions are detected with
    :func:`easybfe.boresch.utils.find_rotatable_torsions`, tracked along the
    trajectory with
    :func:`easybfe.boresch.utils.compute_torsions_along_trajectory`, and reduced
    to a per-torsion median reference (see
    :func:`easybfe.boresch.utils.median_reference`). The representative frame is
    the one minimizing the mean circular distance to that reference (see
    :func:`easybfe.boresch.utils.compute_frame_deviations`). Optionally the
    protein structure, the ligand structure, a torsion-distribution figure and a
    plain-text selection report of the representative frame are written to disk.

    Parameters
    ----------
    topology : os.PathLike
        Path to the topology file (e.g. a PDB) readable by MDAnalysis.
    trajectory : os.PathLike
        Path to the trajectory file (e.g. an XTC) readable by MDAnalysis.
    ligand_sdf : os.PathLike, optional
        Path to an SDF file describing the ligand. Its atom ordering must match
        the ligand atom ordering in the trajectory. Exactly one of ``ligand_sdf``
        or ``ligand_mol`` must be given.
    ligand_mol : rdkit.Chem.Mol, optional
        Ligand molecule (with explicit hydrogens) whose atom ordering matches the
        trajectory. Exactly one of ``ligand_sdf`` or ``ligand_mol`` must be given.
    ligand_selection : str, optional
        MDAnalysis selection string for the ligand atoms. When ``None`` the first
        ``N`` atoms of the system are used as the ligand, where ``N`` is the number
        of atoms in the ligand. Default is ``None``.
    protein_selection : str, optional
        MDAnalysis selection string for the protein atoms written to ``out_pdb``.
        Default is ``'protein'``.
    out_pdb : os.PathLike, optional
        If given, write the protein structure of the representative frame here.
    out_sdf : os.PathLike, optional
        If given, write the ligand structure of the representative frame here.
    out_fig : os.PathLike, optional
        If given, write the torsion-distribution figure here.
    out_info : os.PathLike, optional
        If given, write the plain-text selection report (per-torsion means and
        representative-frame values) here.
    topology_format : str, optional
        Topology format passed to MDAnalysis. If ``None`` it is inferred.
    trajectory_format : str, optional
        Trajectory format passed to MDAnalysis. If ``None`` it is inferred.

    Returns
    -------
    dict
        Dictionary with the keys

        ``rep_frame``
            Index of the representative frame (:class:`int`).
        ``deviations``
            Per-frame circular deviation, shape ``(n_frames,)``.
        ``torsions``
            Dihedral values in radians, shape ``(n_frames, M)``.
        ``reference``
            Per-torsion median reference in radians, shape ``(M,)``.
        ``rep_values``
            Torsion values at the representative frame in radians, shape ``(M,)``.
        ``torsion_atoms``
            List of torsion atom-index 4-tuples.
        ``info``
            The plain-text selection report (:class:`str`).

    Raises
    ------
    ValueError
        If neither or both of ``ligand_sdf`` / ``ligand_mol`` are given, if the
        ligand has no rotatable torsions, or if the resolved ligand selection does
        not contain exactly ``N`` atoms.
    """
    import MDAnalysis as mda
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    if (ligand_sdf is None) == (ligand_mol is None):
        raise ValueError("Exactly one of 'ligand_sdf' or 'ligand_mol' must be given.")

    if ligand_mol is None:
        ligand_mol = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)[0]
        if ligand_mol is None:
            raise ValueError(f"Could not parse ligand SDF: {ligand_sdf}")
    n_ligand_atoms = ligand_mol.GetNumAtoms()

    torsion_atoms = find_rotatable_torsions(ligand_mol)
    if not torsion_atoms:
        raise ValueError("No rotatable torsions found in the ligand.")
    logger.info(
        "Representative-frame selection: found %d rotatable torsion(s) in the ligand.",
        len(torsion_atoms),
    )

    universe = mda.Universe(
        topology, trajectory, topology_format=topology_format, format=trajectory_format
    )

    if ligand_selection is None:
        ligand_atoms = universe.atoms[:n_ligand_atoms]
    else:
        ligand_atoms = universe.select_atoms(ligand_selection)
    if ligand_atoms.n_atoms != n_ligand_atoms:
        raise ValueError(
            f"Ligand selection resolved to {ligand_atoms.n_atoms} atoms but the ligand "
            f"has {n_ligand_atoms} atoms; the atom ordering would be inconsistent."
        )
    ligand_indices = ligand_atoms.indices

    torsions = compute_torsions_along_trajectory(universe, ligand_indices, torsion_atoms)
    reference = median_reference(torsions, axis=0)
    deviations = compute_frame_deviations(torsions, reference)
    rep_frame = int(np.argmin(deviations))
    rep_values = torsions[rep_frame]

    info = _format_selection_info(rep_frame, deviations, reference, rep_values, torsion_atoms)
    logger.info(
        "Selected representative frame %d / %d (deviation %.4f).\n%s",
        rep_frame,
        deviations.shape[0],
        float(deviations[rep_frame]),
        info,
    )

    universe.trajectory[rep_frame]

    if out_pdb:
        protein_atoms = universe.select_atoms(protein_selection)
        if protein_atoms.n_atoms == 0:
            raise ValueError(f"Protein selection '{protein_selection}' selected no atoms.")
        protein_atoms.write(str(out_pdb))
        logger.info("Wrote representative protein structure to %s", out_pdb)

    if out_sdf:
        rep_coords = ligand_atoms.positions
        mol_out = Chem.Mol(ligand_mol)
        conf = mol_out.GetConformer()
        for i in range(n_ligand_atoms):
            x, y, z = rep_coords[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        with Chem.SDWriter(str(out_sdf)) as writer:
            writer.write(mol_out)
        logger.info("Wrote representative ligand structure to %s", out_sdf)

    if out_fig:
        import matplotlib.pyplot as plt

        fig = plot_torsion_distributions(
            torsions, reference, rep_values, rep_frame, torsion_atoms, save_path=out_fig
        )
        plt.close(fig)
        logger.info("Wrote torsion-distribution figure to %s", out_fig)

    if out_info:
        with open(out_info, "w") as handle:
            handle.write(info + "\n")
        logger.info("Wrote representative-frame selection report to %s", out_info)

    return {
        "rep_frame": rep_frame,
        "deviations": deviations,
        "torsions": torsions,
        "reference": reference,
        "rep_values": rep_values,
        "torsion_atoms": torsion_atoms,
        "info": info,
    }
