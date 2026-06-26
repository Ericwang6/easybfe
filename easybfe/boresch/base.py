"""Abstract base class and registry for Boresch restraint finders."""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import Protein, Ligand
from ..core.registry import Registry
from .restraint import BoreschRestraint


logger = logging.getLogger(__name__)


class BoreschRestraintsFinder(ABC):
    """Abstract base class for finding Boresch restraint anchors and building :class:`BoreschRestraint`.

    Parameters
    ----------
    protein : easybfe.core.Protein
        Protein providing the protein anchor atoms.
    ligand : easybfe.core.Ligand
        Ligand providing the ligand anchor atoms.
    wts : tuple of float, optional
        Six Boresch force constants (bond, two angles, three dihedrals).
    workdir : os.PathLike, optional
        Working directory in which a finder may store intermediate and result
        files (structures, plots, selection reports, ...). It is created if it
        does not exist. Concrete finders are free to ignore it; only
        :class:`easybfe.boresch.md_finder.RxRxMDBoreschRestraintsFinder` currently
        writes results here. Default is ``None`` (no files are written).
    """

    def __init__(
        self,
        protein: Protein,
        ligand: Ligand,
        wts: tuple[float, float, float, float, float, float] = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        workdir: Optional[os.PathLike] = None,
        *args,
        **kwargs,
    ):
        self.protein = protein
        self.ligand = ligand
        self.wts = wts
        self.workdir = Path(workdir) if workdir is not None else None
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def find(
        self,
        protein_positions: Optional[np.ndarray] = None,
        ligand_positions: Optional[np.ndarray] = None,
    ) -> BoreschRestraint:
        """Find anchor atoms and return a :class:`BoreschRestraint`.

        If ``protein_positions`` or ``ligand_positions`` are not provided, positions
        are taken from :attr:`protein` and :attr:`ligand` (e.g. from OpenMM/RDKit).
        """
        ...


# Registry for BoreschRestraintsFinder implementations
BORESCH_FINDER_REGISTRY: Registry[BoreschRestraintsFinder] = Registry(BoreschRestraintsFinder)
