"""
Registry for small molecule force field parameterizers.

Parametrizer classes register themselves via the decorator when their
modules are imported (e.g. :class:`easybfe.smff.gaff.GAFF` in :mod:`easybfe.smff.gaff`).
"""
from __future__ import annotations

from ..core.registry import Registry
from .base import SmallMoleculeForceField

PARAMETRIZER_REGISTRY: Registry[SmallMoleculeForceField] = Registry(SmallMoleculeForceField)
