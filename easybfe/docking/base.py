"""Abstract base class for molecular docking engines."""

import abc
import os
import tempfile
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import numpy as np
from rdkit import Chem


def compute_box_from_coordinates(coords: np.ndarray):
    """Compute a docking box that encloses the given coordinates.

    The box is padded by 10 Angstrom on each side and clamped to a
    minimum of 25 Angstrom per dimension.

    Parameters
    ----------
    coords : numpy.ndarray
        Atom coordinates of shape ``(N, 3)``.

    Returns
    -------
    center : numpy.ndarray
        ``(3,)`` array with the box centre.
    box : numpy.ndarray
        ``(3,)`` array with the box dimensions.
    """
    vmax = np.max(coords, axis=0)
    vmin = np.min(coords, axis=0)
    center = (vmax + vmin) / 2
    box = vmax - vmin + 10.0
    box = np.where(box < 25.0, 25.0, box)
    return center, box


class BaseDocking(abc.ABC):
    """Abstract base for docking backends.

    Handles protein path validation, docking-box setup (either from
    explicit coordinates or from a reference molecule), and working-
    directory management.  Subclasses must implement :meth:`dock`,
    :meth:`rescore`, and :meth:`constr_dock`.

    Parameters
    ----------
    protein : os.PathLike
        Path to the protein structure file.
    box_center : tuple of float, optional
        ``(x, y, z)`` centre of the docking box in Angstrom.
        Required unless *ref_mol* is given.
    box_size : tuple of float, optional
        ``(x, y, z)`` dimensions of the docking box in Angstrom.
        Required unless *ref_mol* is given.
    ref_mol : :class:`rdkit.Chem.Mol`, optional
        Reference molecule whose conformer coordinates are used to
        compute *box_center* and *box_size* via
        :func:`compute_box_from_coordinates`.  Ignored when both
        *box_center* and *box_size* are provided explicitly.
    wdir : os.PathLike, optional
        Working directory for intermediate files.  A temporary directory
        is created when *None*.

    Raises
    ------
    FileNotFoundError
        If *protein* does not exist.
    ValueError
        If neither explicit box parameters nor *ref_mol* are provided.
    """

    def __init__(
        self,
        protein: os.PathLike,
        *,
        box_center: Optional[Tuple[float, float, float]] = None,
        box_size: Optional[Tuple[float, float, float]] = None,
        ref_mol: Optional[Chem.Mol] = None,
        wdir: Optional[os.PathLike] = None,
    ):
        protein = Path(protein)
        if not protein.is_file():
            raise FileNotFoundError(f'{protein} does not exist')
        self.protein_input = protein.resolve()

        if box_center is not None and box_size is not None:
            self.box_center = list(box_center)
            self.box_size = list(box_size)
        elif ref_mol is not None:
            center, size = compute_box_from_coordinates(
                ref_mol.GetConformer().GetPositions()
            )
            self.box_center = center.tolist()
            self.box_size = size.tolist()
        else:
            raise ValueError(
                "Either (box_center, box_size) or ref_mol must be provided"
            )

        if wdir is not None:
            self.wdir = Path(wdir).resolve()
            self.wdir.mkdir(parents=True, exist_ok=True)
        else:
            self._tmp_wdir = tempfile.TemporaryDirectory(prefix='docking_')
            self.wdir = Path(self._tmp_wdir.name)

    @abc.abstractmethod
    def dock(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """Run docking and return ranked poses."""
        ...

    @abc.abstractmethod
    def rescore(self, mol: Chem.Mol) -> float:
        """Score an existing ligand pose."""
        ...

    @abc.abstractmethod
    def constr_dock(
        self,
        mol: Chem.Mol,
        ref_mol: Chem.Mol,
        mapping: Optional[Dict[int, int]] = None,
        **kwargs,
    ) -> Chem.Mol:
        """Constrained (local) docking / optimisation of a pose."""
        ...
