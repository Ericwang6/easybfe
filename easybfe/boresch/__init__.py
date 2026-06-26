"""Boresch restraint selection for absolute binding free-energy calculations.

This package locates Boresch anchor atoms (three protein, three ligand), builds
the six restraint degrees of freedom, and computes the analytical free-energy
correction. The submodules are:

- :mod:`easybfe.boresch.utils`: geometry, circular statistics, atom-mapping and
  rotatable-torsion helpers.
- :mod:`easybfe.boresch.restraint`: the :class:`BoreschRestraint` container and
  :func:`compute_boresch_energy`.
- :mod:`easybfe.boresch.base`: the :class:`BoreschRestraintsFinder` base class
  and the :data:`BORESCH_FINDER_REGISTRY`.
- :mod:`easybfe.boresch.finders`: single-structure finders (``'rxrx'`` and
  ``'user'``).
- :mod:`easybfe.boresch.md_finder`: the trajectory-based finder (``'rxrx-md'``).
- :mod:`easybfe.boresch.select_rep`: representative-frame selection from a
  complex trajectory.
"""

from .utils import (
    compute_bond,
    compute_angle,
    compute_dihedral,
    _bond_series,
    _angle_series,
    _dihedral_series,
    circular_mean,
    median_reference,
    _circular_mean_deg,
    _circular_std_rad,
    _map_ligand_atom_to_candidate,
    _enumerate_backbone_candidates,
    draw_ligand_anchors,
    find_rotatable_torsions,
    compute_torsions_along_trajectory,
    compute_frame_deviations,
    plot_torsion_distributions,
)
from .restraint import BoreschRestraint, compute_boresch_energy
from .base import BoreschRestraintsFinder, BORESCH_FINDER_REGISTRY
from .finders import RxRxBoreschRestraintsFinder, UserSpecifiedBoreschRestraint
from .md_finder import RxRxMDBoreschRestraintsFinder
from .select_rep import select_representative_frame

__all__ = [
    "compute_bond",
    "compute_angle",
    "compute_dihedral",
    "draw_ligand_anchors",
    "circular_mean",
    "median_reference",
    "find_rotatable_torsions",
    "compute_torsions_along_trajectory",
    "compute_frame_deviations",
    "plot_torsion_distributions",
    "select_representative_frame",
    "BoreschRestraint",
    "compute_boresch_energy",
    "BoreschRestraintsFinder",
    "BORESCH_FINDER_REGISTRY",
    "RxRxBoreschRestraintsFinder",
    "UserSpecifiedBoreschRestraint",
    "RxRxMDBoreschRestraintsFinder",
]
