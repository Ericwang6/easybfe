'''
Author: Eric Wang
Date: 10/07/2024

This file contains base class for atom mapping
'''
import os
import abc
import json
from copy import deepcopy
from typing import Dict
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from IPython.display import Image as IpythonImage


def get_sc_atoms_and_bonds(mol, cc_list):
    atoms, bonds = [], []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in cc_list:
            atoms.append(atom.GetIdx())
            for bo in atom.GetBonds():
                if bo.GetIdx() not in bonds:
                    bonds.append(bo.GetIdx())
    return atoms, bonds


def draw_atom_mapping(ligA: Chem.Mol, ligB: Chem.Mol, mapping: Dict[int, int]):
    ligandA, ligandB = deepcopy(ligA), deepcopy(ligB)
    AllChem.Compute2DCoords(ligandA)
    AllChem.Compute2DCoords(ligandB)
    AllChem.AlignMol(ligandA, ligandB, atomMap=list(mapping.items()))
    
    sc_atoms_A, sc_bonds_A = get_sc_atoms_and_bonds(ligandA, mapping.keys())
    sc_atoms_B, sc_bonds_B = get_sc_atoms_and_bonds(ligandB, mapping.values())

    return Draw.MolsToGridImage(
        [ligandA, ligandB],
        molsPerRow=2,
        subImgSize=(500, 500),
        legends=[ligandA.GetProp('_Name'), ligandB.GetProp('_Name')],
        highlightAtomLists=[sc_atoms_A, sc_atoms_B],
        highlightBondLists=[sc_bonds_A, sc_bonds_B]
    )


class LigandRbfeAtomMapper(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...
    
    @abc.abstractmethod
    def run_mapping(self, ligandA: Chem.Mol, ligandB: Chem.Mol) -> Dict[int, int]:
        ...
    
    def run(self, ligandA: Chem.Mol | os.PathLike, ligandB: Chem.Mol | os.PathLike, wdir: os.PathLike) -> Dict[int, int]:
        if not isinstance(ligandA, Chem.Mol):
            ligandA = Chem.SDMolSupplier(str(ligandA), removeHs=False)[0]
        if not isinstance(ligandB, Chem.Mol):
            ligandB = Chem.SDMolSupplier(str(ligandB), removeHs=False)[0]

        mapping = self.run_mapping(ligandA, ligandB)
        mapping = dict(list(sorted([(k, v) for k, v in mapping.items()], key=lambda x: x[0])))
        with open(os.path.join(wdir, 'atom_mapping.json'), 'w') as f:
            json.dump(mapping, f, indent=4)
        img = draw_atom_mapping(ligandA, ligandB, mapping)
        if isinstance(img, IpythonImage):
            with open(os.path.join(wdir, 'atom_mapping.png'), 'wb') as f:
                f.write(img.data)
        else:
            img.save(os.path.join(wdir, 'atom_mapping.png'))
        return mapping