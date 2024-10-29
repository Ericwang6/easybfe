'''
Author: Eric Wang
Date: 10/07/2024

This file contains OpenFE-implemented atom mapping algorithms.
'''
from typing import Dict
from rdkit import Chem
from openfe.setup import LomapAtomMapper
from kartograf import KartografAtomMapper

from .base import LigandRbfeAtomMapper


class OpenFEAtomMapper(LigandRbfeAtomMapper):
    """
    Wrapper for OpenFE-implemented atom mapping algorithms: Lomap and Kartograf
    """
    def __init__(self, method='lomap', **kwargs):
        if method == 'lomap':
            self.mapper = LomapAtomMapper(**kwargs)
        elif method == 'kartograf':
            # Build Kartograf Atom Mapper
            self.mapper = KartografAtomMapper(**kwargs)
        else:
            raise NotImplementedError(f'Unsupported mapping method: {method}')
    
    def run_mapping(self, ligandA: Chem.Mol, ligandB: Chem.Mol) -> Dict[int, int]:
        from openfe import SmallMoleculeComponent

        mapping_candidate = next(self.mapper.suggest_mappings(
            SmallMoleculeComponent(ligandA),
            SmallMoleculeComponent(ligandB)
        )).componentA_to_componentB
        
        # I find sometimes it will map two atoms with different elements, which is not desired
        mapping = {}
        for k, v in mapping_candidate.items():
            if ligandA.GetAtomWithIdx(k).GetSymbol() != ligandB.GetAtomWithIdx(v).GetSymbol():
                continue
            mapping[k] = v
        return mapping