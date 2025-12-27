'''
Author: Eric Wang
Date: 12/13/2024

This file contains wrappers for OpenFE-implemented atom mapping algorithms: Lomap and Kartograf.
'''
from typing import Dict
from rdkit import Chem
import openfe.setup as setup
import warnings
from .base import LigandRbfeAtomMapper


class OpenFEAtomMapper(LigandRbfeAtomMapper):
    """
    Wrapper for OpenFE-implemented atom mapping algorithms: Lomap and Kartograf
    """
    def __init__(
        self, 
        method='lomap', 
        allow_element_change: bool = False, 
        allow_map_hydrogen_to_non_hydrogen: bool = False, 
        allow_hybridization_change: bool = False, 
        **kwargs
    ):
        if method == 'lomap':
            self.mapper = setup.LomapAtomMapper(**kwargs)
        elif method == 'kartograf':
            # Build Kartograf Atom Mapper
            self.mapper = setup.KartografAtomMapper(**kwargs)
        else:
            raise NotImplementedError(f'Unsupported mapping method: {method}')
        
        self.allow_element_change = allow_element_change
        self.allow_map_hydrogen_to_non_hydrogen = allow_map_hydrogen_to_non_hydrogen
        if (not self.allow_element_change) and self.allow_map_hydrogen_to_non_hydrogen:
            self.allow_map_hydrogen_to_non_hydrogen = False
            warnings.warn('allow_element_change is False, so allow_map_hydrogen_to_non_hydrogen is forcibly set to False')
        self.allow_hybridization_change = allow_hybridization_change
    
    def run_mapping(self, ligandA: Chem.Mol, ligandB: Chem.Mol) -> Dict[int, int]:
        from openfe import SmallMoleculeComponent

        mapping_candidate = next(self.mapper.suggest_mappings(
            SmallMoleculeComponent(ligandA),
            SmallMoleculeComponent(ligandB)
        )).componentA_to_componentB
        
        mapping = {}
        for k, v in mapping_candidate.items():
            atom_a = ligandA.GetAtomWithIdx(k)
            atom_b = ligandB.GetAtomWithIdx(v)
            if atom_a.GetSymbol() != atom_b.GetSymbol():
                is_element_change = True
                if atom_a == 'H' or atom_b == 'H':
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
        
        # handle hydrogens, if two atoms are mapped (with same element) and their hydrogens must be mapped
        # if two atoms are not mapped, their hydrogens are not mapped
        # to_pop_out = []
        # to_add = {}
        # if not self.allow_map_hydrogen_to_non_hydrogen:
        #     for k, v in mapping.items():
        #         atom_a = ligandA.GetAtomWithIdx(k)
        #         atom_b = ligandB.GetAtomWithIdx(v)
        #         if atom_a.GetSymbol() == 'H' and atom_b.GetSymbol() == 'H':
        #             parent_a = atom_a.GetNeighbors()[0]
        #             parent_b = atom_b.GetNeighbors()[0]
        #             if mapping.get(parent_a.GetIdx(), -1) != parent_b.GetIdx():
        #                to_pop_out.append(k)
        #         if atom_a.GetSymbol() != 'H' and atom_b.GetSymbol() != 'H':
        #             hydrogens_a = [nbr.GetIdx() for nbr in atom_a.GetNeighbors() if nbr.GetSymbol() == 'H']
        #             hydrogens_b = [nbr.GetIdx() for nbr in atom_b.GetNeighbors() if nbr.GetSymbol() == 'H']
        #             min_len = min(len(hydrogens_a), len(hydrogens_b))
        #             for i in range(min_len):
        #                 to_add[hydrogens_a[i]] = hydrogens_b[i]
                
        # for k in to_pop_out:
        #     mapping.pop(k)

        return mapping