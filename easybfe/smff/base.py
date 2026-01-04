'''
Author: Eric Wang
Date: 10/07/2024

This file contains abstract class for a small molecule force field parameterizer
'''
import abc
import os
from typing import Optional


class SmallMoleculeForceField(abc.ABC):
    """
    Abstract base class for small molecule force field parameterizers.
    
    This class defines the interface that all force field parameterization
    implementations must follow. Subclasses should implement methods to
    generate force field parameters for small molecules in various formats
    
    Examples
    --------
    Subclasses include :class:`GAFF`, :class:`OpenFF`, and :class:`CustomForceField`.
    """
    
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the force field parameterizer.
        
        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        ...

    @abc.abstractmethod
    def parametrize(self, ligand_file: os.PathLike, wdir: Optional[os.PathLike] = None):
        """
        Generate force field parameters for a ligand molecule.
        
        This method should generate force field parameter files (e.g., prmtop,
        inpcrd, top, xml) for the input ligand structure.
        
        Parameters
        ----------
        ligand_file : os.PathLike
            Path to the input ligand structure file. Supported formats depend
            on the implementation (typically SDF, MOL, or MOL2).
        wdir : os.PathLike, optional
            Working directory where output files will be written. If None,
            files are written to the current directory.
        
        Notes
        -----
        The output files are typically named based on the stem of the input
        ligand file. For example, if `ligand_file` is "molecule.sdf", the
        output files might be "molecule.prmtop", "molecule.inpcrd", etc.
        """
        ...




