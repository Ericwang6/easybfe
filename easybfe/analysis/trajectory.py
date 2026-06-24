from __future__ import annotations

import os, shutil, warnings
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import capped_distance, minimize_vectors
from MDAnalysis.transformations import wrap, unwrap, center_in_box
try:
    from MDAnalysis.guesser.tables import SYMB2Z
except ModuleNotFoundError:
    from MDAnalysis.topology.tables import SYMB2Z

import spyrmsd.rmsd

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes


def find_dihedrals(mol, redundant=True):
    dihe_patt = Chem.MolFromSmarts('[!#1]!#[*]-;!@[*]!#[!#1]')
    if redundant:
        dihes = list(mol.GetSubstructMatches(dihe_patt))
    else:
        dihes = {}
        for match in mol.GetSubstructMatches(dihe_patt):
            rb = (match[1], match[2])
            if (rb in dihes or (rb[1], rb[0]) in dihes):
                continue
            dihes[rb] = match 
        dihes = list(dihes.values())
    dihes.sort(key=lambda x: (x[1], x[2]))
    return dihes


def plot_dihe(ax, dihedrals, bin_width=15):
    bins = np.arange(-180, 180 + bin_width, bin_width)
    ax.hist(dihedrals, density=True, range=(-180, 180), bins=bins, edgecolor='black')
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-6, 7) * 30)


def plot_dihe_with_mol(u: mda.Universe, mol: Chem.Mol, save_dir: os.PathLike, bin_width=15):
    dihes = find_dihedrals(mol)
    diheObj = Dihedral([u.atoms[list(indices)] for indices in dihes])
    diheObj.run()

    mol_noh = Chem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol_noh)
    heavy_atom_mapping = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        heavy_atom_mapping[atom.GetIdx()] = len(heavy_atom_mapping)

    for i, dihe in tqdm(enumerate(dihes), total=len(dihes), desc='Plotting torsion'):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        atoms = [heavy_atom_mapping[idx] for idx in dihe]
        bonds = [
            mol_noh.GetBondBetweenAtoms(atoms[0], atoms[1]).GetIdx(),
            mol_noh.GetBondBetweenAtoms(atoms[1], atoms[2]).GetIdx(),
            mol_noh.GetBondBetweenAtoms(atoms[2], atoms[3]).GetIdx(),
        ]
        img = Draw.MolToImage(mol_noh, size=(1000, 500), highlightAtoms=atoms, highlightBonds=bonds)
        axes[0].imshow(img)
        axes[0].axis('off')
        plot_dihe(axes[1], diheObj.angles[:, i], bin_width)
        fig.savefig(os.path.join(save_dir, f'torsion_{"-".join([str(x) for x in dihe])}.png'), dpi=300)
        plt.close(fig)


def compute_rmsd_naive(u: mda.Universe, ligand_str: str, save_path: os.PathLike = ""):
    ligand = u.select_atoms(ligand_str)
    ref_pos = ligand.positions.copy()
    start_time = u.trajectory[0].time
    rmsd_list = []
    time_list = []
    for ts in u.trajectory:
        rmsd = np.sqrt(np.sum((ligand.positions - ref_pos) ** 2) / ref_pos.shape[0])
        rmsd_list.append(rmsd)
        time_list.append((ts.time - start_time) / 1000)
    
    data = np.array([time_list, rmsd_list]).T
    if save_path:
        np.savetxt(save_path, data)

    return data


def _nearest_water_residues(
    water_atoms: mda.AtomGroup,
    target_atoms: mda.AtomGroup,
    count: int,
    box: Optional[np.ndarray],
    initial_cutoff: float = 5.0,
) -> mda.AtomGroup:
    """Select complete water residues with the smallest atom-atom distance to a target."""
    if count <= 0:
        return water_atoms[:0]
    if target_atoms.n_atoms == 0:
        raise ValueError("include_water_selection selected no atoms")

    cutoff = initial_cutoff
    residue_distances: dict[int, float] = {}
    while len(residue_distances) < count:
        pairs, distances = capped_distance(
            target_atoms.positions,
            water_atoms.positions,
            max_cutoff=cutoff,
            box=box,
            return_distances=True,
        )
        residue_distances.clear()
        if pairs.size:
            water_resindices = water_atoms.resindices[pairs[:, 1]]
            for resindex, distance in zip(water_resindices, distances):
                residue_distances[resindex] = min(
                    residue_distances.get(int(resindex), np.inf),
                    float(distance),
                )

        if len(residue_distances) >= count:
            break
        cutoff *= 1.5
        if cutoff > 1000:
            raise RuntimeError(f"Could not find {count} water residues near the requested selection")

    selected_resindices = [
        resindex
        for resindex, _ in sorted(residue_distances.items(), key=lambda item: item[1])[:count]
    ]
    selected = water_atoms.universe.atoms[:0]
    for resindex in selected_resindices:
        selected += water_atoms.universe.residues[resindex].atoms
    return selected


def _count_nearby_water_residues(
    water_atoms: mda.AtomGroup,
    target_atoms: mda.AtomGroup,
    box: Optional[np.ndarray],
    cutoff: float = 5.0,
) -> int:
    if target_atoms.n_atoms == 0:
        raise ValueError("include_water_selection selected no atoms")
    pairs = capped_distance(
        target_atoms.positions,
        water_atoms.positions,
        max_cutoff=cutoff,
        box=box,
        return_distances=False,
    )
    if not pairs.size:
        return 0
    return len(np.unique(water_atoms.resindices[pairs[:, 1]]))


def _shift_waters_to_target_image(
    waters: mda.AtomGroup,
    target_atoms: mda.AtomGroup,
    box: Optional[np.ndarray],
) -> None:
    """Translate each water residue into the periodic image nearest the target.

    With ``process_pbc=False`` the raw coordinates of a retained water can lie in a
    periodic image far from ``target_atoms`` even when it is the closest water under the
    minimum image convention (waters diffuse freely and may cross into an adjacent image
    box). Molecular viewers do not apply the minimum image convention, so such a water
    would be rendered far from the solute. This translates every water residue by an
    integer combination of box vectors so that its raw coordinates sit in the same image
    as the nearest atom of ``target_atoms`` (``include_water_selection``), making the raw
    water-target distance equal to the minimum image distance.

    The positions are modified in place on the current timestep. Each water residue is
    moved as a whole (rigid body), so internal geometry is preserved.

    Parameters
    ----------
    waters : MDAnalysis.core.groups.AtomGroup
        Complete water residues to reposition.
    target_atoms : MDAnalysis.core.groups.AtomGroup
        Atoms of ``include_water_selection`` that define the reference image.
    box : numpy.ndarray, optional
        Periodic box dimensions ``[lx, ly, lz, alpha, beta, gamma]``. When ``None`` no
        shift is applied (a non-periodic system cannot have image artefacts).
    """
    if box is None or waters.n_atoms == 0 or target_atoms.n_atoms == 0:
        return

    target_positions = target_atoms.positions
    for residue in waters.residues:
        atoms = residue.atoms
        # Representative point: the oxygen (heaviest atom) of the rigid water residue.
        ref_local = int(np.argmax(atoms.masses))
        ref_pos = atoms.positions[ref_local]
        # Vector from each target atom to the water reference atom, and its minimum image.
        deltas = ref_pos - target_positions
        min_deltas = minimize_vectors(deltas, box=box)
        # Closest target atom under the minimum image convention.
        nearest = int(np.argmin(np.einsum('ij,ij->i', min_deltas, min_deltas)))
        # Integer box-vector shift that places the water in that target atom's image.
        shift = min_deltas[nearest] - deltas[nearest]
        if np.any(np.abs(shift) > 1e-6):
            atoms.positions = atoms.positions + shift


def _regularize_water_records(pdb_path: os.PathLike, output_atoms: mda.AtomGroup) -> None:
    """
    Tidy retained-water records in a written PDB.

    For every retained water this (1) keeps only the two O-H bonds as ``CONECT`` records
    (regenerated from the residue's oxygen and hydrogens, dropping any other water-
    involving connectivity), and (2) sets the water chain ID to ``X`` (or ``Y`` when
    ``X`` is already used by a non-water chain) and renumbers the water residues
    sequentially from 1. The original solvent residue numbers from the full system
    overflow the 4-column PDB ``resSeq`` field; renumbering avoids that.

    Parameters
    ----------
    pdb_path : os.PathLike
        Path to the PDB file to rewrite in place.
    output_atoms : MDAnalysis.core.groups.AtomGroup
        Atoms written to ``pdb_path``, in the same order, used to locate waters and
        their oxygen/hydrogen serials.
    """
    water = output_atoms.select_atoms("water")
    if water.n_atoms == 0:
        return

    # Choose a water chain ID not already used by the non-water atoms (protein,
    # ligand, ions): prefer ``X``, fall back to ``Y`` and then other letters.
    non_water = output_atoms.select_atoms("not water")
    used_chains: set[str] = set()
    if hasattr(non_water, "chainIDs"):
        used_chains.update(str(c).strip() for c in non_water.chainIDs if str(c).strip())
    water_chain = "X"
    if water_chain in used_chains:
        for candidate in ("Y",) + tuple(chr(c) for c in range(ord("A"), ord("Z") + 1)):
            if candidate not in used_chains:
                water_chain = candidate
                break

    water_index_set = {int(idx) for idx in water.indices}
    serial_to_newresid: dict[int, int] = {}
    water_serials: set[int] = set()
    oh_bonds: list[tuple[int, list[int]]] = []

    cur_resindex = None
    cur_serials: list[int] = []
    cur_masses: list[float] = []
    new_resid = 0

    def _flush() -> None:
        nonlocal new_resid, cur_serials, cur_masses
        if not cur_serials:
            return
        new_resid += 1
        o_local = int(np.argmax(cur_masses))
        o_serial = cur_serials[o_local]
        h_serials = [s for k, s in enumerate(cur_serials) if k != o_local]
        oh_bonds.append((o_serial, h_serials))
        for s in cur_serials:
            serial_to_newresid[s] = new_resid
            water_serials.add(s)
        cur_serials = []
        cur_masses = []

    for i, atom in enumerate(output_atoms):
        if atom.index not in water_index_set:
            continue
        serial = i + 1
        if cur_resindex is not None and atom.resindex != cur_resindex:
            _flush()
        cur_resindex = atom.resindex
        cur_serials.append(serial)
        cur_masses.append(float(atom.mass))
    _flush()

    path = Path(pdb_path)
    body_lines: list[str] = []
    end_line: Optional[str] = None
    for line in path.read_text().splitlines():
        rec = line[:6]
        if rec in ("ATOM  ", "HETATM"):
            serial = int(line[6:11])
            if serial in water_serials:
                resid = serial_to_newresid[serial]
                line = f"{line[:21]}{water_chain}{resid:>4d}{line[26:]}"
            body_lines.append(line)
        elif rec == "CONECT":
            serials = [
                int(line[k:k + 5])
                for k in range(6, len(line.rstrip()), 5)
                if line[k:k + 5].strip()
            ]
            # Drop any water-involving connectivity; O-H bonds are regenerated below.
            if any(s in water_serials for s in serials):
                continue
            body_lines.append(line)
        elif line.rstrip() == "END":
            end_line = line
        else:
            body_lines.append(line)

    for o_serial, h_serials in oh_bonds:
        body_lines.append("CONECT" + f"{o_serial:>5d}" + "".join(f"{h:>5d}" for h in h_serials))
        for h in h_serials:
            body_lines.append("CONECT" + f"{h:>5d}" + f"{o_serial:>5d}")

    body_lines.append(end_line if end_line is not None else "END")
    path.write_text("\n".join(body_lines) + "\n")


def post_process_trajectory(
    in_top: os.PathLike,
    in_trj: os.PathLike,
    out_pdb: os.PathLike = '',
    out_trj: os.PathLike = '',
    ref: os.PathLike = '',
    process_pbc: bool = False,
    do_alignment: bool = True,
    in_top_format: Optional[str] = None,
    in_trj_format: Optional[str] = None,
    ref_format: Optional[str] = None,
    center_selection: str = 'protein',
    output_selection: str = 'protein or resname MOL',
    align_selection: str = 'backbone',
    remove_tmp: str = True,
    include_water_selection: Optional[str] = None,
    water_distance: float = 5.0,
):
    """
    Post process MD trajectory

    Parameters
    ----------
    in_top : os.PathLike
        Path to topology file
    in_trj : os.PathLike
        Path to trajectory file  
    ref : os.PathLike, optional
        Path to reference structure for alignment. If not provided, use first frame.
        When provided, this reference structure (not the first production frame) is also
        dumped as ``out_pdb`` (the processed topology), so no separate first-frame PDB is
        written.
    out_pdb : os.PathLike, optional
        Path to output PDB file (processed topology). Written from ``ref`` when ``ref`` is
        given, otherwise from the first processed frame.
    out_trj : os.PathLike, optional 
        Path to output trajectory file
    process_pbc : bool, optional
        Whether to process periodic boundary conditions with MDAnalysis. Default is
        ``False``. AMBER never breaks bonds during a simulation, so every molecule
        (solute and each water) stays intact across the trajectory and no wrapping is
        required to keep molecules whole. Retained waters are instead shifted into the
        image of ``include_water_selection`` on a per-residue basis (see
        :func:`easybfe.analysis.trajectory._shift_waters_to_target_image`), which keeps
        their raw coordinates physically adjacent to the target for correct
        visualization. When set to ``True``, the trajectory is additionally centered on
        ``center_selection`` and wrapped into the unit cell.
    do_alignment : bool, optional
        Whether to align trajectory to reference. Default is True
    in_top_format : str, optional
        Format of topology file. If None, format is inferred by MDAnalysis
    in_trj_format : str, optional
        Format of trajectory file. If None, format is inferred by MDAnalysis
    ref_format : str, optional
        Format of reference structure file. If None, format is inferred by MDAnalysis
    center_selection : str, optional
        Selection string for centering. Default is 'protein'
    output_selection : str, optional
        Selection string for output. Default is 'protein or resname MOL'
    include_water_selection : str, optional
        If provided, count water residues within ``water_distance`` of this selection in
        the starting frame. In every frame, retain that many complete water residues
        nearest to the selection in addition to ``output_selection``. The starting frame
        is ``ref`` when provided (loaded with ``in_top``), otherwise the first frame of
        ``in_trj``.
    water_distance : float, optional
        Distance cutoff in Angstrom used to determine the number of retained waters
        from the starting frame. Default is 5.0.
    align_selection : str, optional
        Selection string for alignment. Default is 'backbone'
    remove_tmp : bool, optional
        Whether to remove temporary files. Default is True

    Returns
    -------
    bool
        True if successful
    """

    u = mda.Universe(in_top, in_trj, topology_format=in_top_format, format=in_trj_format)
    if process_pbc:
        transformations = [
            center_in_box(u.select_atoms(center_selection)),
            wrap(u.atoms),
            unwrap(u.atoms)
        ]
        u.trajectory.add_transformations(*transformations)
        desc = f'Output {output_selection} + Processing PBC'
    else:
        desc = f'Output {output_selection}'

    wdir = Path(out_trj).parent
    stem = Path(out_trj).stem
    tmp_trj = str(wdir / f'{stem}_tmp{Path(out_trj).suffix}')
    
    # When a reference structure is given, it (not the first production frame) is dumped
    # as the processed topology PDB, and it defines how many waters (N) are retained.
    use_ref_topology = bool(ref)

    selection = u.select_atoms(output_selection)
    target_atoms = None
    water_atoms = None
    water_count = 0

    # The reference universe (when provided) defines the number of retained waters and is
    # written as the processed topology PDB. Loaded once and reused below.
    ref_universe = None
    ref_target = None
    ref_water_atoms = None
    ref_box = None
    if use_ref_topology:
        ref_universe = mda.Universe(in_top, ref, topology_format=in_top_format, format=ref_format)
        ref_universe.trajectory[0]

    if include_water_selection is not None:
        target_atoms = u.select_atoms(include_water_selection)
        # Candidate waters are all waters except those already part of the target
        # selection (``include_water_selection``), so the target's own waters are never
        # picked as "nearby" waters of themselves.
        target_resindices = set(target_atoms.resindices)
        water_atoms = u.select_atoms("water")
        if target_resindices:
            water_atoms = water_atoms[
                ~np.isin(water_atoms.resindices, np.fromiter(target_resindices, dtype=int))
            ]

        # Production frame-0 box (essentially constant under NPT); used as the PBC box and
        # as a fallback when the reference structure does not expose box dimensions.
        u.trajectory[0]
        traj_box = None if u.trajectory.ts.dimensions is None else u.trajectory.ts.dimensions.copy()

        if use_ref_topology:
            ref_target = ref_universe.select_atoms(include_water_selection)
            ref_target_resindices = set(ref_target.resindices)
            ref_water_atoms = ref_universe.select_atoms("water")
            if ref_target_resindices:
                ref_water_atoms = ref_water_atoms[
                    ~np.isin(ref_water_atoms.resindices, np.fromiter(ref_target_resindices, dtype=int))
                ]
            ref_box = ref_universe.dimensions
            if ref_box is None or np.allclose(ref_box[:3], 1.0):
                ref_box = traj_box
            # N = number of water residues within the cutoff in the reference frame.
            water_count = _count_nearby_water_residues(
                ref_water_atoms, ref_target, ref_box, cutoff=water_distance
            )
        else:
            # No reference: define N from the first production frame.
            water_count = _count_nearby_water_residues(
                water_atoms, target_atoms, traj_box, cutoff=water_distance
            )
        desc += f' + {water_count} nearest waters'

    output_n_atoms = selection.n_atoms
    if include_water_selection is not None and water_count:
        # Water residues have a fixed number of atoms; size the output from one selection.
        sample_waters = _nearest_water_residues(
            water_atoms, target_atoms, water_count, traj_box, initial_cutoff=water_distance
        )
        output_n_atoms += sample_waters.n_atoms

    with mda.Writer(tmp_trj, n_atoms=output_n_atoms) as W:
        for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory), desc=desc):
            output_atoms = selection
            if include_water_selection is not None and water_count:
                # Pick the N nearest waters in THIS frame (by distance, not by identity,
                # since waters diffuse) using the minimum image convention.
                selected_waters = _nearest_water_residues(
                    water_atoms,
                    target_atoms,
                    water_count,
                    ts.dimensions,
                    initial_cutoff=water_distance,
                )
                # The raw coordinates of a selected water may lie in an adjacent image
                # box; shift each water residue into the target's image so the dumped
                # (no minimum image convention) coordinates stay adjacent to the target.
                _shift_waters_to_target_image(selected_waters, target_atoms, ts.dimensions)
                output_atoms = selection + selected_waters
            if output_atoms.n_atoms != output_n_atoms:
                raise RuntimeError(
                    f"Selected {output_atoms.n_atoms} atoms, expected {output_n_atoms}; "
                    "water residues may have inconsistent atom counts"
                )
            W.write(output_atoms)
            if i == 0 and not use_ref_topology:
                output_atoms.write(out_pdb)
                _regularize_water_records(out_pdb, output_atoms)

    if use_ref_topology:
        # Processed topology PDB: output_selection plus the N nearest waters in the
        # reference frame, all at reference coordinates.
        ref_output = ref_universe.select_atoms(output_selection)
        if include_water_selection is not None and water_count:
            ref_waters = _nearest_water_residues(
                ref_water_atoms, ref_target, water_count, ref_box, initial_cutoff=water_distance
            )
            _shift_waters_to_target_image(ref_waters, ref_target, ref_box)
            ref_output = ref_output + ref_waters
        ref_output.write(out_pdb)
        _regularize_water_records(out_pdb, ref_output)

    if do_alignment:
        if use_ref_topology:
            u_ref = mda.Universe(out_pdb)
        else:
            u_ref = mda.Universe(out_pdb, tmp_trj)

        u_tmp = mda.Universe(out_pdb, tmp_trj)
        alignment = align.AlignTraj(u_tmp, u_ref, select=align_selection, in_memory=True)
        alignment.run()

        with mda.Writer(out_trj, n_atoms=u_tmp.atoms.n_atoms) as W:
            for i, ts in tqdm(enumerate(u_tmp.trajectory), total=len(u_tmp.trajectory), desc='Align'):
                W.write(u_tmp.atoms)
        
        if remove_tmp:
            os.remove(tmp_trj)

    else:
        shutil.move(tmp_trj, out_trj)

    return True


def compute_rmsd(
    top: os.PathLike,
    trj: os.PathLike,
    ref: os.PathLike = '',
    selection: str = 'resname MOL',
    use_symmetry_correction: bool = True,
    heavy_atoms_only: bool = True,
    top_format: Optional[str] = None,
    trj_format: Optional[str] = None,
    ref_format: Optional[str] = None,
    ref_top: Optional[os.PathLike] = None,
    ref_top_format: Optional[str] = None,
    use_ref_as_first_frame: bool = True,
    save_path: os.PathLike = ''
):
    """
    Calculate RMSD between trajectory and reference structure.

    Parameters
    ----------
    top : os.PathLike
        Path to topology file
    trj : os.PathLike 
        Path to trajectory file
    ref : os.PathLike, optional
        Path to reference structure. If not provided, use first frame as reference
    selection : str, optional
        Selection string for atoms to compute RMSD on. Default is 'resname MOL'
    use_symmetry_correction : bool, optional
        Whether to use symmetry correction (powered by spyrmsd package) in RMSD calculation. Default is True.
    heavy_atoms_only : bool, optional
        Whether to only use heavy atoms for RMSD calculation. Default is True.
    top_format : str, optional
        Format of topology file. If None, format is inferred by MDAnalysis. Default is None.
    trj_format : str, optional
        Format of trajectory file. If None, format is inferred by MDAnalysis. Default is None.
    ref_format : str, optional
        Format of reference structure file. If None, format is inferred by MDAnalysis. Default is None.
    ref_top : os.PathLike, optional
        Topology used to read ``ref`` when the reference comes from a different structure
        than the trajectory (e.g. the full starting system ``<basename>.pdb`` while ``top``
        is the selection-only processed PDB). ``selection`` must match the same atoms in
        both topologies. When None, ``top`` is used as the reference topology. Default is None.
    ref_top_format : str, optional
        Format of ``ref_top``. If None, format is inferred by MDAnalysis. Default is None.
    use_ref_as_first_frame: bool, optional
        Wheter to include the reference structure (if provided) as the first frame when report the results. Default is True.
    save_path : os.PathLike, optional
        Path to save RMSD data

    Returns
    -------
    numpy.ndarray
        Array of shape (n_frames, 2) containing time (in nanosecond) and RMSD values (in Angstrom)
    """
    
    u = mda.Universe(top, trj, topology_format=top_format, format=trj_format, to_guess=('types', 'masses', 'bonds'))

    atoms = u.select_atoms(selection)
    atomic_nums = np.array([SYMB2Z[a] for a in atoms.elements], dtype=np.int32)
    heavy_mask = [i for i in range(atoms.n_atoms) if atomic_nums[i] > 1] if heavy_atoms_only else list(range(atoms.n_atoms))

    # reference structure
    if not ref:
        u.trajectory[0]
        coords_ref = atoms.positions.copy()[heavy_mask]
    else:
        # Reference may use its own topology (e.g. the full starting structure) as long
        # as `selection` resolves to the same atoms/order as in the trajectory.
        ref_topology = ref_top if ref_top is not None else top
        ref_topology_format = ref_top_format if ref_top is not None else top_format
        u_ref = mda.Universe(ref_topology, ref, topology_format=ref_topology_format, format=ref_format, to_guess=('types', 'masses', 'bonds'))
        atoms_ref = u_ref.select_atoms(selection)
        if atoms_ref.n_atoms != atoms.n_atoms:
            raise ValueError(
                f"Selection '{selection}' matches {atoms.n_atoms} atoms in the trajectory "
                f"but {atoms_ref.n_atoms} in the reference structure."
            )
        u_ref.trajectory[0]
        coords_ref = atoms_ref.positions.copy()[heavy_mask]

    # coordinates read from trajectory
    time_list = []
    coords = []
    for i in range(len(u.trajectory)):
        ts = u.trajectory[i]
        time_list.append(ts.time / 1000) # time unit in nanoseconds
        coords.append(atoms.positions[heavy_mask])
    
    time_list = np.array(time_list)
    dt_list = time_list[1:] - time_list[:-1]
    dt = np.mean(dt_list)
    if np.any((dt_list - dt) > 1e-5):
        warnings.warn(f"Non-uniform dt detected, will use the mean val {dt:.3e} ns as dt")
    time_list -= time_list[0] - dt 

    # calculate rmsd
    if use_symmetry_correction:
        atom_indices_map = {atoms.indices[i]: i for i in range(atoms.n_atoms)}
        bonds = np.array(
            [
                [atom_indices_map[b[0]], atom_indices_map[b[1]]]
                for b in atoms.bonds.indices
                if b[0] in atom_indices_map and b[1] in atom_indices_map
            ],
            dtype=np.int32,
        ).reshape(-1, 2).T
        adj = np.zeros((atoms.n_atoms, atoms.n_atoms), dtype=np.int32)
        if bonds.size:
            adj[bonds[0], bonds[1]] = 1
            adj[bonds[1], bonds[0]] = 1
        atomic_nums = atomic_nums[heavy_mask]
        adj = adj[heavy_mask][:, heavy_mask]
        rmsd_list = spyrmsd.rmsd.symmrmsd(coords_ref, coords, atomic_nums, atomic_nums, adj, adj)
    else:
        rmsd_list = [np.sqrt(np.sum((c - coords_ref) ** 2) / coords_ref.shape[0]) for c in coords]

    data = np.array([time_list, rmsd_list]).T
    if save_path:
        np.savetxt(save_path, data, header='Time (ns), RMSD (angstrom)')
    
    return data


def plot_rmsd(
    data: Union[os.PathLike, np.ndarray],
    name: str = '',
    save_path: str = '',
    ax: Optional[Axes] = None,
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(5, 4))
    if not isinstance(data, np.ndarray):
        data = np.loadtxt(data, comments=['#'])

    time_list, rmsd_list = data[:, 0], data[:, 1]
    title = f'{name} Average RMSD: {np.mean(rmsd_list):.2f} \u00C5'

    ax.plot(time_list, rmsd_list)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('RMSD (\u00C5)')
    ax.set_xlim(0, time_list[-1])
    ax.set_ylim(0)
    ax.set_title(title.strip())
    if save_path:
        ax.figure.savefig(save_path, **kwargs)
    
    return ax
