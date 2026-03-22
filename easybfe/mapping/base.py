'''
Author: Eric Wang
Date: 10/07/2024

This file contains base class for atom mapping
'''
import os
import abc
import json
from copy import deepcopy
from typing import Union, Optional, Any
import warnings
from pathlib import Path
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from IPython.display import Image as IpythonImage

from ..core import Ligand


logger = logging.getLogger(__name__)


def get_sc_atoms_and_bonds(mol, cc_list):
    atoms, bonds = [], []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in cc_list:
            atoms.append(atom.GetIdx())
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() not in cc_list and bond.GetEndAtomIdx() not in cc_list:
            bonds.append(bond.GetIdx())
    return atoms, bonds


def draw_atom_mapping(ligA: Chem.Mol, ligB: Chem.Mol, mapping: dict[int, int]):
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

    def __init__(
        self,  
        *,
        allow_element_change: bool = True,
        allow_map_hydrogen_to_non_hydrogen: bool = True,
        allow_hybridization_change: bool = True
    ):
        self.allow_element_change = allow_element_change
        self.allow_map_hydrogen_to_non_hydrogen = allow_map_hydrogen_to_non_hydrogen
        if (not self.allow_element_change) and self.allow_map_hydrogen_to_non_hydrogen:
            self.allow_map_hydrogen_to_non_hydrogen = False
            warnings.warn('allow_element_change is False, so allow_map_hydrogen_to_non_hydrogen is forcibly set to False')
        self.allow_hybridization_change = allow_hybridization_change
    
    @abc.abstractmethod
    def propose_mapping(self, ligandA: Ligand, ligandB: Ligand) -> dict[int, int]:
        ...
    
    def post_process_mapping(self, ligandA: Chem.Mol, ligandB: Chem.Mol, mapping_candidate: dict[int, int]) -> dict[int, int]:
        mapping = {}
        for k, v in mapping_candidate.items():
            atom_a = ligandA.GetAtomWithIdx(k)
            atom_b = ligandB.GetAtomWithIdx(v)
            if atom_a.GetSymbol() != atom_b.GetSymbol():
                is_element_change = True
                if atom_a.GetSymbol() == 'H' or atom_b.GetSymbol() == 'H':
                    is_hydrogen_to_non_hydrogen = True
                else:
                    is_hydrogen_to_non_hydrogen = False
            else:
                is_element_change = False
                is_hydrogen_to_non_hydrogen = False
            
            if is_element_change and (not self.allow_element_change):
                continue
            if is_hydrogen_to_non_hydrogen and (not self.allow_map_hydrogen_to_non_hydrogen):
                continue
            if (atom_a.GetHybridization() != atom_b.GetHybridization()) and (not self.allow_hybridization_change):
                continue
            mapping[k] = v
        
        # if H1 and H2 are mapped but their parent are not, remove them
        to_pop = []
        for k, v in mapping.items():
            atom_a = ligandA.GetAtomWithIdx(k)
            atom_b = ligandB.GetAtomWithIdx(v)
            if atom_a.GetSymbol() == 'H' and atom_b.GetSymbol() == 'H':
                parent_a = atom_a.GetNeighbors()[0]
                parent_b = atom_b.GetNeighbors()[0]
                if parent_b.GetIdx() != mapping.get(parent_a.GetIdx(), None):
                    to_pop.append(k)
        
        for k in to_pop:
            mapping.pop(k)
        return mapping
    
    def run(self, ligandA: Ligand, ligandB: Ligand, wdir: os.PathLike) -> dict[int, int]:
        molA, molB = ligandA.get_rdmol(), ligandB.get_rdmol()
        mapping = self.post_process_mapping(
            molA, molB, self.propose_mapping(ligandA, ligandB)
        )
        mapping = dict(list(sorted([(k, v) for k, v in mapping.items()], key=lambda x: x[0])))
        with open(os.path.join(wdir, 'atom_mapping.json'), 'w') as f:
            json.dump(mapping, f, indent=4)
        img = draw_atom_mapping(molA, molB, mapping)
        if isinstance(img, IpythonImage):
            with open(os.path.join(wdir, 'atom_mapping.png'), 'wb') as f:
                f.write(img.data)
        else:
            img.save(os.path.join(wdir, 'atom_mapping.png'))
        return mapping


class CustomLigandAtomMapper(LigandRbfeAtomMapper):
    def __init__(self, data: Union[dict, str, Path], **kwargs):
        super().__init__(**kwargs)
        if isinstance(data, Path) or isinstance(data, str):
            with open(data) as f:
                _data = json.load(f)
        else:
            _data = data
        
        assert isinstance(_data, dict), 'Content must be a dictionary'
        try:
            self._mapping = {int(k): int(v) for k, v in _data.values()}
            self._mode = 'single'
        except ValueError as e:
            self._mapping = {}
            for k, v in _data.items():
                pert = k.split('~')
                self._mapping[(pert[0], pert[1])] = {int(vk): int(vv) for vk, vv in v.items()}
            self._mode = 'multiple'
    
    def propose_mapping(self, ligandA: Ligand, ligandB: Ligand) -> dict[int, int]:
        if self._mode == 'single':
            return self._mapping
        else:
            return self._mapping[(ligandA.name, ligandB.name)]