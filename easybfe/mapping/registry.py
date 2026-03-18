"""
Registry for ligand RBFE atom mappers.

Mapper classes register themselves via the decorator when their
modules are imported (e.g. :class:`easybfe.mapping.lazymcs.LazyMCSMapper` in
:mod:`easybfe.mapping.lazymcs`).
"""
from __future__ import annotations

from ..core.registry import Registry
from .base import LigandRbfeAtomMapper

MAPPER_REGISTRY: Registry[LigandRbfeAtomMapper] = Registry(LigandRbfeAtomMapper)
