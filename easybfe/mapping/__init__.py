"""
Ligand atom mapping for RBFE.

Provides a registry of atom mapper implementations (LazyMCS, Kartograf, Lomap,
custom) and a :func:`load_mapper` helper to instantiate them by name.
"""

import logging
from typing import Any

from .base import CustomLigandAtomMapper, LigandRbfeAtomMapper
from .registry import MAPPER_REGISTRY

MAPPER_REGISTRY.add("custom", CustomLigandAtomMapper)

try:
    from .lazymcs import LazyMCSMapper
except ImportError:
    LazyMCSMapper = None  # type: ignore[misc, assignment]
try:
    from .kartograf import KartografAtomMapper
except ImportError:
    KartografAtomMapper = None  # type: ignore[misc, assignment]
try:
    from .lomap import LomapAtomMapper
except ImportError:
    LomapAtomMapper = None  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)


def load_mapper(name: str, raise_errors: bool = False, **kwargs: Any) -> LigandRbfeAtomMapper:
    """
    Load an atom mapper by registry name.

    Parameters
    ----------
    name : str
        Registry key: ``"lazymcs"``, ``"kartograf"``, ``"lomap"``, or ``"custom"``.
    raise_errors : bool, optional
        If ``True``, exceptions from mapper construction are re-raised without
        logging. Default ``False``.
    **kwargs
        Passed to the mapper constructor. For ``"custom"``, ``data`` is
        required (mapping data or path).

    Returns
    -------
    LigandRbfeAtomMapper
        Instance of the requested mapper.

    Raises
    ------
    NotImplementedError
        If ``name`` is not in the registry. Message includes available names.
    """
    if name not in MAPPER_REGISTRY.names():
        available = ", ".join(MAPPER_REGISTRY.names())
        raise NotImplementedError(
            f"Mapper {name!r} is not available. Available: {available}."
        )

    try:
        if name == "custom":
            data = kwargs.pop("data", None)
            if data is None:
                raise ValueError("load_mapper('custom', ...) requires data=...")
            return MAPPER_REGISTRY.create(name, data, **kwargs)
        return MAPPER_REGISTRY.create(name, **kwargs)
    except Exception as e:
        if not raise_errors:
            logger.exception("Failed to create mapper %r: %s", name, e)
        raise
