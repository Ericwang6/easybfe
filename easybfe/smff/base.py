'''
Author: Eric Wang
Date: 10/07/2024

This file contains abstract class for a small molecule force field parameterizer
'''
import abc
import os
from typing import Optional


class SmallMoleculeForceField(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def parametrize(self, ligand_file: os.PathLike, wdir: Optional[os.PathLike] = None):
        ...




