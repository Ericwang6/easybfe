"""Geometry and atom-mapping helpers for Boresch restraint selection.

This module collects the low-level numerical utilities used to build and score
Boresch restraints: scalar geometry (``compute_bond`` / ``compute_angle`` /
``compute_dihedral``), their numpy-vectorized trajectory counterparts, circular
statistics, the RDKit-based ligand atom / protein backbone mapping helpers, and
the rotatable-torsion utilities used to pick a representative trajectory frame.
"""

import os
import math
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import MDAnalysis as mda
    from matplotlib.figure import Figure
    from rdkit import Chem


def compute_bond(pos0, pos1):
    dx, dy, dz = pos1[0]-pos0[0], pos1[1]-pos0[1], pos1[2]-pos0[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)


def compute_angle(pos0, pos1, pos2):
    """
    Compute the angle between three positions.
    
    Computes the angle at pos1 formed by the vectors pos0->pos1 and pos1->pos2.
    
    Parameters
    ----------
    pos0 : array-like
        First position (x, y, z).
    pos1 : array-like
        Central position (x, y, z).
    pos2 : array-like
        Third position (x, y, z).
    
    Returns
    -------
    float
        Angle in degrees, in the range (0, 180).
    """
    # Convert to numpy arrays for vector operations
    pos0 = np.array(pos0)
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    
    # Compute vectors
    vec1 = pos0 - pos1  # Vector from pos1 to pos0
    vec2 = pos2 - pos1  # Vector from pos1 to pos2
    
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute angle: zero-length vector")
    
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2
    
    # Compute angle using dot product
    cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def compute_dihedral(pos0, pos1, pos2, pos3):
    """
    Compute the dihedral angle (torsion angle) between four positions.
    
    Computes the dihedral angle defined by the four positions pos0-pos1-pos2-pos3.
    This is the angle between the planes defined by pos0-pos1-pos2 and pos1-pos2-pos3.
    
    Parameters
    ----------
    pos0 : array-like
        First position (x, y, z).
    pos1 : array-like
        Second position (x, y, z).
    pos2 : array-like
        Third position (x, y, z).
    pos3 : array-like
        Fourth position (x, y, z).
    
    Returns
    -------
    float
        Dihedral angle in degrees, in the range (-180, 180).
    """
    # Convert to numpy arrays for vector operations
    pos0 = np.array(pos0)
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    pos3 = np.array(pos3)
    
    # Compute vectors
    vec1 = pos1 - pos0  # Vector from pos0 to pos1
    vec2 = pos2 - pos1  # Vector from pos1 to pos2
    vec3 = pos3 - pos2  # Vector from pos2 to pos3
    
    # Compute cross products to get normal vectors to the planes
    cross1 = np.cross(vec1, vec2)  # Normal to plane pos0-pos1-pos2
    cross2 = np.cross(vec2, vec3)  # Normal to plane pos1-pos2-pos3
    
    # Normalize cross products
    norm1 = np.linalg.norm(cross1)
    norm2 = np.linalg.norm(cross2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute dihedral: degenerate geometry (atoms are collinear)")
    
    cross1 = cross1 / norm1
    cross2 = cross2 / norm2
    
    # Normalize vec2 for sign determination
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec2 == 0:
        raise ValueError("Cannot compute dihedral: zero-length central bond")
    vec2_norm = vec2 / norm_vec2
    
    # Compute sin and cos of dihedral angle
    cos_angle = np.clip(np.dot(cross1, cross2), -1.0, 1.0)
    sin_angle = np.dot(np.cross(cross1, cross2), vec2_norm)
    
    # Use atan2 for proper sign determination (returns angle in [-pi, pi])
    dihedral_rad = math.atan2(sin_angle, cos_angle)
    dihedral_deg = math.degrees(dihedral_rad)
    
    return dihedral_deg


def _bond_series(pos0: np.ndarray, pos1: np.ndarray) -> np.ndarray:
    """Distance between two position time series.

    Parameters
    ----------
    pos0, pos1 : numpy.ndarray
        Position arrays of shape ``(F, 3)`` (one row per frame).

    Returns
    -------
    numpy.ndarray
        Distances of shape ``(F,)`` in the input length unit.
    """
    return np.linalg.norm(np.asarray(pos1, dtype=float) - np.asarray(pos0, dtype=float), axis=-1)


def _angle_series(pos0: np.ndarray, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """Angle at ``pos1`` formed by ``pos0`` and ``pos2`` for a time series.

    Parameters
    ----------
    pos0, pos1, pos2 : numpy.ndarray
        Position arrays of shape ``(F, 3)``.

    Returns
    -------
    numpy.ndarray
        Angles in degrees of shape ``(F,)`` in the range ``[0, 180]``.
    """
    vec1 = np.asarray(pos0, dtype=float) - np.asarray(pos1, dtype=float)
    vec2 = np.asarray(pos2, dtype=float) - np.asarray(pos1, dtype=float)
    norm1 = np.linalg.norm(vec1, axis=-1)
    norm2 = np.linalg.norm(vec2, axis=-1)
    denom = norm1 * norm2
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.einsum("ij,ij->i", vec1, vec2) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _dihedral_series(
    pos0: np.ndarray, pos1: np.ndarray, pos2: np.ndarray, pos3: np.ndarray
) -> np.ndarray:
    """Dihedral ``pos0-pos1-pos2-pos3`` for a time series.

    Parameters
    ----------
    pos0, pos1, pos2, pos3 : numpy.ndarray
        Position arrays of shape ``(F, 3)``.

    Returns
    -------
    numpy.ndarray
        Dihedral angles in degrees of shape ``(F,)`` in the range ``(-180, 180]``.
    """
    b1 = np.asarray(pos1, dtype=float) - np.asarray(pos0, dtype=float)
    b2 = np.asarray(pos2, dtype=float) - np.asarray(pos1, dtype=float)
    b3 = np.asarray(pos3, dtype=float) - np.asarray(pos2, dtype=float)
    cross1 = np.cross(b1, b2)
    cross2 = np.cross(b2, b3)
    b2_norm = np.linalg.norm(b2, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        b2_hat = b2 / b2_norm
    x = np.einsum("ij,ij->i", cross1, cross2)
    y = np.einsum("ij,ij->i", np.cross(cross1, cross2), b2_hat)
    return np.degrees(np.arctan2(y, x))


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


def median_reference(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the median of ``values`` as an actual observed sample.

    Unlike :func:`numpy.median`, which averages the two central samples for an
    even-sized population, this returns the lower of the two central samples so
    that the result is always a value that actually occurred in the input. When
    the values are angles this avoids any averaging across the periodic boundary:
    no circular-mean handling is required because the reference is a single
    observed sample rather than a synthetic average (see
    :func:`circular_mean`).

    Parameters
    ----------
    values : numpy.ndarray
        Array of samples.
    axis : int, optional
        Axis along which the median is taken. Default is ``0``.

    Returns
    -------
    numpy.ndarray
        The median sample along ``axis``. For an even population size ``n`` the
        sample at sorted position ``n // 2 - 1`` (the lower central sample) is
        returned; for an odd size the exact central sample is returned.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    # Lower-middle index: exact center for odd n, lower of the two centers for
    # even n. Either way the returned value is an actually observed sample.
    mid = (n - 1) // 2
    partitioned = np.partition(values, mid, axis=axis)
    return np.take(partitioned, mid, axis=axis)


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Circular mean of a set of angles given in degrees.

    Thin wrapper around :func:`circular_mean` that converts to and from
    degrees.

    Parameters
    ----------
    angles_deg : array-like
        Angles in degrees.

    Returns
    -------
    float
        Circular mean in degrees, in the range ``(-180, 180]``.
    """
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    return float(np.degrees(circular_mean(angles)))


def _circular_std_rad(angles_deg: np.ndarray) -> float:
    """Circular standard deviation of angles given in degrees.

    Uses the standard definition :math:`\\sqrt{-2 \\ln R}`, where ``R`` is the
    mean resultant length, returning the result in radians.

    Parameters
    ----------
    angles_deg : array-like
        Angles in degrees.

    Returns
    -------
    float
        Circular standard deviation in radians (``>= 0``).
    """
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    resultant = np.hypot(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    resultant = float(min(max(resultant, 1e-12), 1.0))
    return float(np.sqrt(-2.0 * np.log(resultant)))


def draw_ligand_anchors(
    ligand_mol,
    ligand_anchors,
    out_path: Optional[str] = None,
    size: tuple = (700, 600),
    remove_hs: bool = True,
):
    """Draw the three ligand Boresch anchor atoms on a 2D depiction.

    The three anchors ``(L1, L2, L3)`` are highlighted with distinct colors and
    annotated with ``L1`` / ``L2`` / ``L3`` notes, on top of a freshly computed
    2D depiction of the ligand.

    Parameters
    ----------
    ligand_mol : rdkit.Chem.Mol
        Ligand molecule (typically with explicit hydrogens) whose atom indices
        match ``ligand_anchors``.
    ligand_anchors : sequence of int
        The three ligand anchor atom indices ``(L1, L2, L3)`` in
        ``ligand_mol`` ordering.
    out_path : str, optional
        Path to write the PNG image to. When ``None`` the PNG bytes are
        returned without writing to disk.
    size : tuple of int, optional
        ``(width, height)`` of the output image in pixels. Default
        ``(700, 600)``.
    remove_hs : bool, optional
        Whether to remove explicit hydrogens before drawing (the anchor indices
        are remapped accordingly). Default ``True``.

    Returns
    -------
    bytes
        The PNG image bytes.

    Notes
    -----
    Requires :mod:`rdkit`. The highlight colors are red (``L1``), green
    (``L2``) and blue (``L3``).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    anchors = [int(idx) for idx in ligand_anchors]
    if len(anchors) != 3:
        raise ValueError(f"Expected exactly 3 ligand anchors, got {len(anchors)}")

    mol = Chem.Mol(ligand_mol)
    # Remove non-anchor hydrogens for a cleaner depiction, remapping the anchor
    # indices so they keep pointing to the same atoms.
    if remove_hs:
        to_remove = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 1 and atom.GetIdx() not in anchors
        ]
        editable = Chem.RWMol(mol)
        for idx in sorted(to_remove, reverse=True):
            editable.RemoveAtom(idx)
        mol = editable.GetMol()
        kept = [a for a in range(ligand_mol.GetNumAtoms()) if a not in to_remove]
        old_to_new = {old: new for new, old in enumerate(kept)}
        anchors = [old_to_new[a] for a in anchors]

    Chem.SanitizeMol(mol)
    AllChem.Compute2DCoords(mol)

    labels = ("L1", "L2", "L3")
    colors = {
        anchors[0]: (0.95, 0.40, 0.40),
        anchors[1]: (0.40, 0.80, 0.40),
        anchors[2]: (0.40, 0.55, 0.95),
    }
    for anchor, label in zip(anchors, labels):
        mol.GetAtomWithIdx(anchor).SetProp("atomNote", label)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()
    options.addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=anchors,
        highlightAtomColors=colors,
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    if out_path is not None:
        with open(out_path, "wb") as handle:
            handle.write(png)
    return png


def _map_ligand_atom_to_candidate(ligand_mol, lig_atom_idx: int) -> Optional[int]:
    """Map an interacting ligand atom to a nonterminal heavy candidate atom.

    Following Wu et al. 2025 (map the ligand part of an interaction to the
    candidate atoms) and Chen et al. 2023 (terminal interacting groups are
    reassigned to the bonded nonterminal heavy atom, e.g. a carboxylate oxygen is
    assigned to its carbon), this returns a heavy atom bonded to at least two
    other heavy atoms.

    Parameters
    ----------
    ligand_mol : rdkit.Chem.Mol
        Ligand molecule with explicit hydrogens.
    lig_atom_idx : int
        Index of the interacting ligand atom (0-based, ``ligand_mol`` ordering).

    Returns
    -------
    int or None
        Index of the candidate heavy atom, or ``None`` when no suitable
        nonterminal heavy atom can be reached.
    """

    def _heavy_neighbors(atom):
        return [nei for nei in atom.GetNeighbors() if nei.GetAtomicNum() > 1]

    atom = ligand_mol.GetAtomWithIdx(int(lig_atom_idx))
    if atom.GetAtomicNum() == 1:
        heavy = _heavy_neighbors(atom)
        if not heavy:
            return None
        atom = heavy[0]
    if len(_heavy_neighbors(atom)) >= 2:
        return int(atom.GetIdx())
    for nei in _heavy_neighbors(atom):
        if len(_heavy_neighbors(nei)) >= 2:
            return int(nei.GetIdx())
    return None


def _enumerate_backbone_candidates(residue, l1: int, protein_atoms, ligand_mol):
    """Enumerate six-atom Boresch candidates for one residue and ligand anchor.

    The protein anchors are the backbone ``(CA, C, N)`` atoms of ``residue``
    (mapped to ``P1, P2, P3``), and the ligand anchors are ``l1`` plus a heavy
    neighbor ``l2`` and a heavy next-neighbor ``l3``. Unlike the single-frame
    finder, no instantaneous angle pre-filter is applied here; the trajectory
    angle filter is applied later.

    Parameters
    ----------
    residue : openmm.app.topology.Residue
        Protein residue providing the backbone anchor atoms.
    l1 : int
        Ligand anchor atom index (0-based, ``ligand_mol`` ordering).
    protein_atoms : list of openmm.app.topology.Atom
        All protein atoms, indexable by global atom index.
    ligand_mol : rdkit.Chem.Mol
        Ligand molecule with explicit hydrogens.

    Returns
    -------
    list of tuple
        ``(bb_ca, bb_c, bb_n, l1, l2, l3)`` tuples. Empty when the residue lacks
        the required backbone atoms or no valid ligand neighbors exist.
    """
    bb_ca_atoms = [at for at in residue.atoms() if at.name == "CA"]
    bb_c_atoms = [at for at in residue.atoms() if at.name == "C"]
    if not bb_ca_atoms or not bb_c_atoms:
        return []
    bb_ca = bb_ca_atoms[0].index
    bb_c = bb_c_atoms[0].index
    bb_n = None
    for bo in residue.bonds():
        if bo.atom1 is protein_atoms[bb_c] and bo.atom2.name == "N":
            bb_n = bo.atom2.index
        elif bo.atom2 is protein_atoms[bb_c] and bo.atom1.name == "N":
            bb_n = bo.atom1.index
    if bb_n is None:
        return []

    ligand_atom = ligand_mol.GetAtomWithIdx(int(l1))
    candidates = []
    for nei in ligand_atom.GetNeighbors():
        if nei.GetSymbol() == "H":
            continue
        for nnei in nei.GetNeighbors():
            if nnei.GetSymbol() != "H" and nnei.GetIdx() != l1:
                candidates.append((bb_ca, bb_c, bb_n, int(l1), nei.GetIdx(), nnei.GetIdx()))
    return candidates


# ---------------------------------------------------------------------------
# Rotatable-torsion utilities (representative-frame selection)
# ---------------------------------------------------------------------------
# RDKit SMARTS for a rotatable bond: a single, acyclic bond between two
# non-terminal atoms that are not part of a triple bond. This is the same
# definition RDKit uses internally for ``NumRotatableBonds``. It is compiled
# lazily so that importing this module does not force an RDKit import.
_ROTATABLE_BOND_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
_rotatable_bond_query = None


def find_rotatable_torsions(mol: "Chem.Mol") -> List[Tuple[int, int, int, int]]:
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
    from rdkit import Chem

    global _rotatable_bond_query
    if _rotatable_bond_query is None:
        _rotatable_bond_query = Chem.MolFromSmarts(_ROTATABLE_BOND_SMARTS)

    matches = mol.GetSubstructMatches(_rotatable_bond_query)

    def _pick_terminal(center_idx: int, exclude_idx: int) -> Optional[int]:
        """Pick a heavy neighbor of ``center_idx`` other than ``exclude_idx``.

        Returns ``None`` if the only available neighbors are hydrogens.
        """
        center = mol.GetAtomWithIdx(center_idx)
        heavy = [
            nei.GetIdx()
            for nei in center.GetNeighbors()
            if nei.GetIdx() != exclude_idx and nei.GetAtomicNum() > 1
        ]
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


def compute_torsions_along_trajectory(
    universe: "mda.Universe",
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
    from MDAnalysis.analysis.dihedrals import Dihedral

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


def compute_frame_deviations(torsions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute the per-frame circular deviation from a torsion reference.

    For each frame the deviation is the mean over the ``M`` torsions of the
    circular distance ``d = 1 - cos(phi_i - ref_i)``, where ``ref_i`` is the
    reference value of torsion ``i`` (e.g. its per-torsion median, see
    :func:`median_reference`). The circular distance lies in ``[0, 2]`` and
    correctly accounts for the periodicity of the dihedral angles.

    Parameters
    ----------
    torsions : numpy.ndarray
        Array of shape ``(n_frames, M)`` with dihedral values in radians.
    reference : numpy.ndarray
        Array of shape ``(M,)`` with the reference value of each torsion in
        radians.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_frames,)`` with the mean circular deviation per frame.
    """
    torsions = np.asarray(torsions, dtype=float)
    reference = np.asarray(reference, dtype=float)
    distances = 1.0 - np.cos(torsions - reference[np.newaxis, :])
    return np.mean(distances, axis=1)


def plot_torsion_distributions(
    torsions: np.ndarray,
    reference: np.ndarray,
    rep_values: np.ndarray,
    rep_frame: int,
    torsion_atoms: Sequence[Tuple[int, int, int, int]],
    save_path: Optional[os.PathLike] = None,
    bin_width: float = 15.0,
) -> "Figure":
    """Plot the distribution of every torsion with its reference and representative value.

    One histogram (bar plot) is drawn per torsion showing the distribution of its
    dihedral values (in degrees) across the trajectory. The reference value (the
    per-torsion median, see :func:`median_reference`) and the value at the
    representative frame are highlighted with vertical lines.

    Parameters
    ----------
    torsions : numpy.ndarray
        Array of shape ``(n_frames, M)`` with dihedral values in radians.
    reference : numpy.ndarray
        Array of shape ``(M,)`` with the reference (median) value of each torsion
        in radians.
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
    import matplotlib.pyplot as plt

    torsions_deg = np.rad2deg(np.asarray(torsions, dtype=float))
    reference_deg = np.rad2deg(np.asarray(reference, dtype=float))
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
            reference_deg[k],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"median {reference_deg[k]:.1f}\u00b0",
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
