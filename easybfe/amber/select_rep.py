"""Select a representative frame from a protein-ligand complex trajectory.

The representative frame is the trajectory frame whose ligand rotatable-torsion
geometry is closest to the trajectory-averaged conformation. Closeness is measured
with a circular distance against the per-torsion circular means, so that the
periodic nature of dihedral angles is handled correctly (see
`Circular mean <https://en.wikipedia.org/wiki/Circular_mean>`_).
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict, Any, Sequence, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral

from rdkit import Chem
from rdkit.Geometry import Point3D

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# RDKit SMARTS for a rotatable bond: a single, acyclic bond between two non-terminal
# atoms that are not part of a triple bond. Same definition RDKit uses internally for
# ``NumRotatableBonds``.
_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def find_rotatable_torsions(mol: Chem.Mol) -> List[Tuple[int, int, int, int]]:
    """Find one rotatable torsion per rotatable bond of a molecule.

    The logic is to first find rotatable bonds (single, acyclic, non-terminal),
    and for each bond ``(b, c)`` define a single torsion ``(a, b, c, d)`` where
    ``a`` is a neighbor of ``b`` and ``d`` is a neighbor of ``c``. A heavy
    (non-hydrogen) neighbor is always preferred; a torsion is excluded if either
    terminal would have to be a hydrogen (e.g. methyl rotations).

    Parameters
    ----------
    mol : :class:`rdkit.Chem.Mol`
        Input molecule. Hydrogens should be explicit so that heavy-atom terminals
        can be distinguished from hydrogen terminals.

    Returns
    -------
    list of tuple of int
        List of 4-atom-index tuples ``(a, b, c, d)``, one per retained rotatable
        bond. Indices refer to the atom ordering of ``mol``. The list is sorted by
        the central bond ``(b, c)`` for determinism.
    """
    matches = mol.GetSubstructMatches(_ROTATABLE_BOND_SMARTS)

    def _pick_terminal(center_idx: int, exclude_idx: int) -> Optional[int]:
        """Pick a neighbor of ``center_idx`` other than ``exclude_idx``.

        Heavy atoms are preferred over hydrogens. Returns ``None`` if the only
        available neighbors are hydrogens.
        """
        center = mol.GetAtomWithIdx(center_idx)
        heavy = []
        for nei in center.GetNeighbors():
            if nei.GetIdx() == exclude_idx:
                continue
            if nei.GetAtomicNum() > 1:
                heavy.append(nei.GetIdx())
        if not heavy:
            return None
        return min(heavy)

    torsions: List[Tuple[int, int, int, int]] = []
    seen_bonds = set()
    for b, c in matches:
        bond_key = (min(b, c), max(b, c))
        if bond_key in seen_bonds:
            continue
        a = _pick_terminal(b, c)
        d = _pick_terminal(c, b)
        if a is None or d is None:
            continue
        seen_bonds.add(bond_key)
        torsions.append((a, b, c, d))

    torsions.sort(key=lambda x: (x[1], x[2]))
    return torsions


def circular_mean(angles: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the circular mean of angles in radians.

    The circular mean is defined as
    ``atan2(mean(sin(angles)), mean(cos(angles)))`` and lies in ``(-pi, pi]``.
    See `Circular mean <https://en.wikipedia.org/wiki/Circular_mean>`_.

    Parameters
    ----------
    angles : numpy.ndarray
        Array of angles in radians.
    axis : int, optional
        Axis along which the mean is computed. Default is ``0``.

    Returns
    -------
    numpy.ndarray
        Circular mean in radians, in the range ``(-pi, pi]``.
    """
    angles = np.asarray(angles, dtype=float)
    sin_mean = np.mean(np.sin(angles), axis=axis)
    cos_mean = np.mean(np.cos(angles), axis=axis)
    return np.arctan2(sin_mean, cos_mean)


def compute_torsions_along_trajectory(
    universe: mda.Universe,
    ligand_indices: Sequence[int],
    torsions: Sequence[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Compute dihedral values of ligand torsions over every trajectory frame.

    Parameters
    ----------
    universe : :class:`MDAnalysis.core.universe.Universe`
        Universe holding the trajectory.
    ligand_indices : sequence of int
        Global (universe) atom indices of the ligand atoms, ordered to match the
        atom ordering used by ``torsions`` (i.e. ``ligand_indices[k]`` is the
        global index of the ligand's ``k``-th atom).
    torsions : sequence of tuple of int
        Torsions as 4-tuples of ligand-local atom indices, as returned by
        :func:`find_rotatable_torsions`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames, M)`` with dihedral values in radians, wrapped
        to ``[-pi, pi]``, where ``M`` is the number of torsions.
    """
    ligand_indices = np.asarray(ligand_indices, dtype=int)
    atom_groups = [
        universe.atoms[[ligand_indices[a], ligand_indices[b], ligand_indices[c], ligand_indices[d]]]
        for (a, b, c, d) in torsions
    ]
    dihe = Dihedral(atom_groups).run()
    angles_deg = getattr(dihe, "results", dihe).angles
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=float))
    # Wrap to [-pi, pi] to guarantee a consistent range regardless of the backend.
    return np.arctan2(np.sin(angles_rad), np.cos(angles_rad))


def compute_frame_deviations(torsions: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Compute the per-frame circular deviation from the torsion circular means.

    For each frame the deviation is the mean over the ``M`` torsions of the
    circular distance ``d = 1 - cos(phi_i - mu_i)``, where ``mu_i`` is the
    circular mean of torsion ``i``. This is the distance that the circular mean
    minimizes (see `Circular mean <https://en.wikipedia.org/wiki/Circular_mean>`_),
    and lies in ``[0, 2]``.

    Parameters
    ----------
    torsions : numpy.ndarray
        Array of shape ``(n_frames, M)`` with dihedral values in radians.
    means : numpy.ndarray
        Array of shape ``(M,)`` with the circular mean of each torsion in radians.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames,)`` with the mean circular deviation per frame.
    """
    torsions = np.asarray(torsions, dtype=float)
    means = np.asarray(means, dtype=float)
    distances = 1.0 - np.cos(torsions - means[np.newaxis, :])
    return np.mean(distances, axis=1)


def plot_torsion_distributions(
    torsions: np.ndarray,
    means: np.ndarray,
    rep_values: np.ndarray,
    rep_frame: int,
    torsion_atoms: Sequence[Tuple[int, int, int, int]],
    save_path: Optional[os.PathLike] = None,
    bin_width: float = 15.0,
) -> "Figure":
    """Plot the distribution of every torsion with its mean and representative value.

    One histogram (bar plot) is drawn per torsion showing the distribution of its
    dihedral values (in degrees) across the trajectory. The circular mean and the
    value at the representative frame are highlighted with vertical lines.

    Parameters
    ----------
    torsions : numpy.ndarray
        Array of shape ``(n_frames, M)`` with dihedral values in radians.
    means : numpy.ndarray
        Array of shape ``(M,)`` with the circular mean of each torsion in radians.
    rep_values : numpy.ndarray
        Array of shape ``(M,)`` with the torsion values at the representative
        frame in radians.
    rep_frame : int
        Index of the representative frame (used for annotation).
    torsion_atoms : sequence of tuple of int
        Torsions as 4-tuples of atom indices, used to label each subplot.
    save_path : os.PathLike, optional
        If given, the figure is saved to this path. Default is ``None``.
    bin_width : float, optional
        Histogram bin width in degrees. Default is ``15.0``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    torsions_deg = np.rad2deg(np.asarray(torsions, dtype=float))
    means_deg = np.rad2deg(np.asarray(means, dtype=float))
    rep_deg = np.rad2deg(np.asarray(rep_values, dtype=float))

    n_torsions = torsions_deg.shape[1]
    n_cols = int(np.ceil(np.sqrt(n_torsions)))
    n_rows = int(np.ceil(n_torsions / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )
    bins = np.arange(-180, 180 + bin_width, bin_width)

    for k in range(n_torsions):
        ax = axes[k // n_cols][k % n_cols]
        ax.hist(
            torsions_deg[:, k],
            bins=bins,
            range=(-180, 180),
            edgecolor="black",
            alpha=0.7,
        )
        ax.axvline(
            means_deg[k],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"mean {means_deg[k]:.1f}\u00b0",
        )
        ax.axvline(
            rep_deg[k],
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"rep {rep_deg[k]:.1f}\u00b0",
        )
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-3, 4) * 60)
        ax.set_xlabel("Dihedral (\u00b0)")
        ax.set_ylabel("Count")
        ax.set_title("-".join(str(i) for i in torsion_atoms[k]))
        ax.legend(fontsize=8)

    # Hide any unused axes in the grid.
    for k in range(n_torsions, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle(f"Ligand torsion distributions (representative frame: {rep_frame})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig


def select_representative_frame(
    topology: os.PathLike,
    trajectory: os.PathLike,
    ligand_sdf: os.PathLike,
    ligand_selection: Optional[str] = None,
    protein_selection: str = "protein",
    out_pdb: os.PathLike = "",
    out_sdf: os.PathLike = "",
    out_fig: os.PathLike = "",
    topology_format: Optional[str] = None,
    trajectory_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Select the representative frame of a complex trajectory by ligand torsions.

    The ligand rotatable torsions are detected from ``ligand_sdf``, tracked along
    the trajectory, and reduced to a per-torsion circular mean. The representative
    frame is the one minimizing the mean circular distance to those means (see
    :func:`compute_frame_deviations`). Optionally the protein structure, the ligand
    structure and a torsion-distribution figure of the representative frame are
    written to disk.

    Parameters
    ----------
    topology : os.PathLike
        Path to the topology file (e.g. a PDB) readable by MDAnalysis.
    trajectory : os.PathLike
        Path to the trajectory file (e.g. an XTC) readable by MDAnalysis.
    ligand_sdf : os.PathLike
        Path to an SDF file describing the ligand. Its atom ordering must match
        the ligand atom ordering in the trajectory.
    ligand_selection : str, optional
        MDAnalysis selection string for the ligand atoms. When ``None`` the first
        ``N`` atoms of the system are used as the ligand, where ``N`` is the number
        of atoms in ``ligand_sdf``. Default is ``None``.
    protein_selection : str, optional
        MDAnalysis selection string for the protein atoms written to ``out_pdb``.
        Default is ``'protein'``.
    out_pdb : os.PathLike, optional
        If given, write the protein structure of the representative frame here.
    out_sdf : os.PathLike, optional
        If given, write the ligand structure of the representative frame here.
    out_fig : os.PathLike, optional
        If given, write the torsion-distribution figure here.
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
        ``means``
            Per-torsion circular means in radians, shape ``(M,)``.
        ``torsion_atoms``
            List of torsion atom-index 4-tuples.

    Raises
    ------
    ValueError
        If the ligand has no rotatable torsions, or if the resolved ligand
        selection does not contain exactly ``N`` atoms.
    """
    mol = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)[0]
    if mol is None:
        raise ValueError(f"Could not parse ligand SDF: {ligand_sdf}")
    n_ligand_atoms = mol.GetNumAtoms()

    torsion_atoms = find_rotatable_torsions(mol)
    if not torsion_atoms:
        raise ValueError("No rotatable torsions found in the ligand.")

    universe = mda.Universe(
        topology, trajectory, topology_format=topology_format, format=trajectory_format
    )

    if ligand_selection is None:
        ligand_atoms = universe.atoms[:n_ligand_atoms]
    else:
        ligand_atoms = universe.select_atoms(ligand_selection)
    if ligand_atoms.n_atoms != n_ligand_atoms:
        raise ValueError(
            f"Ligand selection resolved to {ligand_atoms.n_atoms} atoms but the SDF "
            f"has {n_ligand_atoms} atoms; the atom ordering would be inconsistent."
        )
    ligand_indices = ligand_atoms.indices

    torsions = compute_torsions_along_trajectory(universe, ligand_indices, torsion_atoms)
    means = circular_mean(torsions, axis=0)
    deviations = compute_frame_deviations(torsions, means)
    rep_frame = int(np.argmin(deviations))
    rep_values = torsions[rep_frame]

    universe.trajectory[rep_frame]

    if out_pdb:
        protein_atoms = universe.select_atoms(protein_selection)
        if protein_atoms.n_atoms == 0:
            raise ValueError(f"Protein selection '{protein_selection}' selected no atoms.")
        protein_atoms.write(str(out_pdb))

    if out_sdf:
        rep_coords = ligand_atoms.positions
        mol_out = Chem.Mol(mol)
        conf = mol_out.GetConformer()
        for i in range(n_ligand_atoms):
            x, y, z = rep_coords[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        with Chem.SDWriter(str(out_sdf)) as writer:
            writer.write(mol_out)

    if out_fig:
        fig = plot_torsion_distributions(
            torsions, means, rep_values, rep_frame, torsion_atoms, save_path=out_fig
        )
        plt.close(fig)

    return {
        "rep_frame": rep_frame,
        "deviations": deviations,
        "torsions": torsions,
        "means": means,
        "torsion_atoms": torsion_atoms,
    }
