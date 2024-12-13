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
    def __init__(self, method='lomap', allow_element_change: bool = False, allow_map_hydrogen_to_non_hydrogen: bool = False, **kwargs):
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
    
    def run_mapping(self, ligandA: Chem.Mol, ligandB: Chem.Mol) -> Dict[int, int]:
        from openfe import SmallMoleculeComponent

        mapping_candidate = next(self.mapper.suggest_mappings(
            SmallMoleculeComponent(ligandA),
            SmallMoleculeComponent(ligandB)
        )).componentA_to_componentB
        
        # I find sometimes it will map two atoms with different elements, which is not desired
        mapping = {}
        for k, v in mapping_candidate.items():
            atom_a = ligandA.GetAtomWithIdx(k).GetSymbol() 
            atom_b = ligandB.GetAtomWithIdx(v).GetSymbol()
            if atom_a != atom_b:
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
            mapping[k] = v
        return mapping