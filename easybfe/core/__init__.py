"""
Core data models and utilities for EasyBFE.

This module provides the core data structures for ligands, proteins, perturbations,
and database models used throughout the EasyBFE package.
"""

from .ligand import Ligand, LigandLoader
from .protein import Protein
# from .perturbation import LigandPerturbation
# from .sql_models import Base, ProteinDB, LigandDB, LigandPerturbationDB
# from .errors import ResourceConflictError, ResourceNotFoundError

# __all__ = [
#     # Ligand models
#     'Ligand',
#     'LigandLoader',
#     # Protein models
#     'Protein',
#     # Perturbation models
#     'LigandPerturbation',
#     # Database models
#     'Base',
#     'ProteinDB',
#     'LigandDB',
#     'LigandPerturbationDB',
#     # Errors
#     'ResourceConflictError',
#     'ResourceNotFoundError'
# ]
