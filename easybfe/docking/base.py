'''
Author: Eric Wang
Date: 10/07/2024

This file contains abstract class for docking
'''
import abc
import os
import numpy as np
from typing import Optional, List, Any, Dict
from pathlib import Path



class BaseDocking(abc.ABC):
    def __init__(self, protein_pdb: os.PathLike, docking_box: List[float], wdir: os.PathLike, **kwargs):
        self.docking_box = docking_box
        self.wdir = Path(wdir).resolve()
        self.wdir.mkdir(exist_ok=True)
        self.protein_file = self.prepare_protein(protein_pdb)
        self.config = kwargs
    
    @abc.abstractmethod
    def prepare_protein(protein_pdb: os.PathLike):
        ...

    @abc.abstractmethod
    def dock(self, ligand):
        ...
    
    @abc.abstractmethod
    def rescore(self, ligand):
        ...


def compute_box_from_coordinates(coords: np.ndarray):
    vmax = np.max(coords, axis=0)
    vmin = np.min(coords, axis=0)
    center = (vmax + vmin) / 2
    box = vmax - vmin + 10.0
    box = np.where(box < 25.0, 25.0, box)
    return center, box
    