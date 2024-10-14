"""
Author: Eric Wang
Date: 10/07/2024

This file contains a lazy implementation to generate atom mapping between two molecules with rdkit MCS search
"""
import os
from typing import Dict
import numpy as np
from scipy.spatial.distance import cdist
from rdkit import Chem
try:
    from rdkit.Chem.rdFMCS import FindMCS
except:
    from rdkit.Chem.MCS import FindMCS

from .base import LigandRbfeAtomMapper


def find_mcs(molA, molB):
    molA_noh = Chem.RemoveHs(molA)
    molB_noh = Chem.RemoveHs(molB)
    
    mcs = FindMCS([molA_noh, molB_noh])
    # handle mcs failed
    mcs_failed = mcs.canceled if hasattr(mcs, 'canceled') else False
    mcs_failed = (mcs_failed or mcs.numAtoms == 0)
    if mcs_failed:
        raise RuntimeError("MCS Failed")
        
    mcsSmarts = mcs.smarts if hasattr(mcs, 'smarts') else mcs.smartsString
    
    mcs_struct = Chem.MolFromSmarts(mcsSmarts)
    return mcs_struct


def get_common_core(molA, molB, mcs, use_position_info=True):
    mcs = Chem.RemoveHs(mcs)
    ccA = molA.GetSubstructMatch(mcs)
    ccB = molB.GetSubstructMatch(mcs)
    cc = []
    
    if use_position_info:
        posA = molA.GetConformer().GetPositions()
        posB = molB.GetConformer().GetPositions()
        dist_mat = cdist(posA, posB)
    for indexA, indexB in zip(ccA, ccB):
        atomA = molA.GetAtomWithIdx(indexA)
        atomB = molB.GetAtomWithIdx(indexB)
        atomA_hs = [nei.GetIdx() for nei in atomA.GetNeighbors() if nei.GetSymbol() == 'H']
        atomB_hs = [nei.GetIdx() for nei in atomB.GetNeighbors() if nei.GetSymbol() == 'H']
        
        cc.append((indexA, indexB))

        if use_position_info:
            if len(atomA_hs) < len(atomB_hs):
                for hA in atomA_hs:
                    hB = np.argmin(dist_mat[hA])
                    if hB in atomB_hs:
                        cc.append((hA, hB))
            else:
                for hB in atomB_hs:
                    hA = np.argmin(dist_mat[:, hB])
                    if hA in atomA_hs:
                        cc.append((hA, hB))
        else:
            for hA, hB in zip(atomA_hs, atomB_hs):
                cc.append((hA, hB))
            
    cc.sort(key=lambda x: x[0])
    cc = np.array(cc, dtype=int)
    return cc


class LazyMCSMapper(LigandRbfeAtomMapper):
    """
    A lazy implementation of MCS search with RDKit plus geomtry-based mapping for hydrogens.
    Not fully tested and cannot handle all cases
    """
    def __init__(self, mcs: Chem.Mol, use_positions: bool = True):
        self.mcs = mcs
        self.use_positions = use_positions
        
    def run_mapping(self, ligandA: Chem.Mol, ligandB: Chem.Mol) -> Dict[int, int]:
        if self.mcs is None:
            self.mcs = find_mcs(ligandA, ligandB)
        cc = get_common_core(ligandA, ligandB, self.mcs, use_position_info=self.use_positions)
        mapping = {int(c[0]): int(c[1]) for c in cc}
        return mapping

    