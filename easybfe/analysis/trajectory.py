import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral


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


def compute_rmsd(u: mda.Universe, ligand_str: str, save_path: os.PathLike = ""):
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