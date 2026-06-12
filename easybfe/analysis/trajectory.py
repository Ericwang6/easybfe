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
from MDAnalysis.lib.distances import capped_distance
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


def _remove_water_conect_records(pdb_path: os.PathLike, water_atom_serials: set[int]) -> None:
    """Remove CONECT records that involve any retained water atom."""
    if not water_atom_serials:
        return

    path = Path(pdb_path)
    filtered_lines = []
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("CONECT"):
            serials = {
                int(line[index:index + 5])
                for index in range(6, len(line), 5)
                if line[index:index + 5].strip()
            }
            if serials & water_atom_serials:
                continue
        filtered_lines.append(line)
    path.write_text("".join(filtered_lines))


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
        Path to reference structure for alignment. If not provided, use first frame
    out_pdb : os.PathLike, optional
        Path to output PDB file
    out_trj : os.PathLike, optional 
        Path to output trajectory file
    process_pbc : bool, optional
        Whether to process periodic boundary conditions with MDAnalysis. Default is False
        because Amber will not break molecules apart when dumping MD trajectories. 
        But MD engines like GROMACS will do, so set this to True will help. 
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
        the first frame. In every frame, retain that many complete water residues
        nearest to the selection in addition to ``output_selection``.
    water_distance : float, optional
        Distance cutoff in Angstrom used to determine the number of retained waters
        from the first frame. Default is 5.0.
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
    
    selection = u.select_atoms(output_selection)
    target_atoms = None
    water_atoms = None
    water_count = 0
    if include_water_selection is not None:
        target_atoms = u.select_atoms(include_water_selection)
        base_resindices = set(selection.resindices)
        water_atoms = u.select_atoms("water")
        if base_resindices:
            water_atoms = water_atoms[
                ~np.isin(water_atoms.resindices, np.fromiter(base_resindices, dtype=int))
            ]
        u.trajectory[0]
        water_count = _count_nearby_water_residues(
            water_atoms,
            target_atoms,
            u.trajectory.ts.dimensions,
            cutoff=water_distance,
        )
        desc += f' + {water_count} nearest waters'

    first_output_indices = None
    output_n_atoms = selection.n_atoms
    if include_water_selection is not None and water_count:
        first_waters = _nearest_water_residues(
            water_atoms,
            target_atoms,
            water_count,
            u.trajectory.ts.dimensions,
            initial_cutoff=water_distance,
        )
        output_n_atoms += first_waters.n_atoms

    with mda.Writer(tmp_trj, n_atoms=output_n_atoms) as W:
        for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory), desc=desc):
            output_atoms = selection
            if include_water_selection is not None and water_count:
                selected_waters = _nearest_water_residues(
                    water_atoms,
                    target_atoms,
                    water_count,
                    ts.dimensions,
                    initial_cutoff=water_distance,
                )
                output_atoms = selection + selected_waters
            if output_atoms.n_atoms != output_n_atoms:
                raise RuntimeError(
                    f"Selected {output_atoms.n_atoms} atoms, expected {output_n_atoms}; "
                    "water residues may have inconsistent atom counts"
                )
            W.write(output_atoms)
            if i == 0:
                output_atoms.write(out_pdb)
                water_indices = set(output_atoms.select_atoms("water").indices)
                water_serials = {
                    index + 1
                    for index, atom in enumerate(output_atoms)
                    if atom.index in water_indices
                }
                _remove_water_conect_records(out_pdb, water_serials)
                first_output_indices = output_atoms.indices.copy()
    
    if do_alignment:
        if ref:
            ref_universe = mda.Universe(
                in_top,
                ref,
                topology_format=in_top_format,
                format=ref_format,
            )
            atoms = (
                ref_universe.atoms[first_output_indices]
                if include_water_selection is not None
                else ref_universe.select_atoms(output_selection)
            )
            ref_pdb = str(wdir / f'{stem}_ref.pdb')
            atoms.write(ref_pdb)
            u_ref = mda.Universe(out_pdb, ref_pdb)
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
        u_ref = mda.Universe(top, ref, topology_format=top_format, format=ref_format, to_guess=('types', 'masses', 'bonds')) 
        atoms_ref = u_ref.select_atoms(selection)
        u_ref.trajectory[0]
        coords_ref = atoms_ref.positions.copy()[heavy_mask]

    # coordinates read from trajectory
    start_time = u.trajectory[0].time
    time_list = []
    coords = []
    for i in range(len(u.trajectory)):
        ts = u.trajectory[i]
        time_list.append((ts.time - start_time) / 1000) # time unit in nanoseconds
        coords.append(atoms.positions[heavy_mask])
    
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
    
    if use_ref_as_first_frame and ref:
        rmsd_list.insert(0, 0.0)
        dt = time_list[1] - time_list[0]
        if any(abs(time_list[i+1]-time_list[i]-dt) > 1e-5 for i in range(len(time_list)-1)):
            warnings.warn('Non uniform dt detected in trajectory')
        time_list = [0.0] + [t+dt for t in time_list]

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
    dt = time_list[1] - time_list[0]
    title = f'{name} Average RMSD: {np.mean(rmsd_list):.2f} \u00C5'

    ax.plot(time_list, rmsd_list)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('RMSD (\u00C5)')
    ax.set_xlim(0, time_list[-1]+dt)
    ax.set_title(title.strip())
    if save_path:
        ax.figure.savefig(save_path, **kwargs)
    
    return ax
