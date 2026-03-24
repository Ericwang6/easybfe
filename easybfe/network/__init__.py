"""
Ligand network generators for RBFE perturbation planning.

This module provides a small registry-backed interface to construct RBFE ligand
networks as edge lists of ligand names. Built-in graph topologies are available
directly, and additional OpenFE-backed planners are conditionally registered
when OpenFE is installed.
"""

from __future__ import annotations

import abc
from collections import Counter
from typing import Any, TYPE_CHECKING

from .registry import NETWORK_GENERATOR_REGISTRY

if TYPE_CHECKING:
    from ..core import Ligand


def _check_unique_names(ligands: list["Ligand"]) -> set[str]:
    """
    Validate ligand names are non-empty and unique.

    Parameters
    ----------
    ligands : list[Ligand]
        Ligands to validate.

    Returns
    -------
    set[str]
        Set of validated ligand names.

    Raises
    ------
    ValueError
        If any ligand name is empty or duplicated.
    """
    names = [lig.name for lig in ligands]
    empty = [i for i, name in enumerate(names) if not str(name).strip()]
    if empty:
        raise ValueError(f"Ligand names must be non-empty. Empty names at indices: {empty}")

    counts = Counter(names)
    duplicated = sorted(name for name, cnt in counts.items() if cnt > 1)
    if duplicated:
        raise ValueError(f"Duplicated ligand names found: {duplicated}")

    return set(names)


def _check_edges(nodes: set[str], edges: list[tuple[str, str]]) -> None:
    """
    Validate network edges against the node set.

    Ensures each edge references valid ligand names and every ligand appears in
    at least one edge.
    """
    node_in_edges: set[str] = set()
    for src, dst in edges:
        assert src in nodes, f"{src} not in ligands"
        assert dst in nodes, f"{dst} not in ligands"
        node_in_edges.add(src)
        node_in_edges.add(dst)
    for node in nodes:
        assert node in node_in_edges, f"{node} not in edges"


class LigandNetworkGenerator(abc.ABC):
    """Abstract interface for RBFE ligand-network construction."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize network generator."""

    @abc.abstractmethod
    def run(self, ligands: list[Ligand]) -> list[tuple[str, str]]:
        """Generate an RBFE network as list of ligand-name edges."""


@NETWORK_GENERATOR_REGISTRY.register("star")
class StarNetworkGenerator(LigandNetworkGenerator):
    """Connect all ligands to a central ligand (star network)."""

    def __init__(self, center: str):
        self.center = center

    def run(self, ligands: list["Ligand"]) -> list[tuple[str, str]]:
        names = _check_unique_names(ligands)
        assert len(names) >= 3, f"Only one or two ligands, {self.__class__.__name__} not applicable"
        assert self.center in names, "Central ligand name not in ligands"
        return [(self.center, name) for name in names if name != self.center]


@NETWORK_GENERATOR_REGISTRY.register("wheel")
class WheelNetworkGenerator(StarNetworkGenerator):
    """Star network plus a cycle among peripheral ligands."""

    def run(self, ligands: list["Ligand"]) -> list[tuple[str, str]]:
        edges = super().run(ligands)
        additional: list[tuple[str, str]] = []
        for i in range(len(edges) - 1):
            additional.append((edges[i][1], edges[i + 1][1]))
        additional.append((edges[-1][1], edges[0][1]))
        return edges + additional


@NETWORK_GENERATOR_REGISTRY.register("bistar")
class BiStarNetworkGenerator(LigandNetworkGenerator):
    """Connect ligands to two hubs with a bridge between hubs."""

    def __init__(self, center1: str, center2: str):
        self.center1 = center1
        self.center2 = center2

    def run(self, ligands: list["Ligand"]) -> list[tuple[str, str]]:
        names = _check_unique_names(ligands)
        assert len(names) >= 3, f"Only one or two ligands, {self.__class__.__name__} not applicable"
        assert self.center1 in names, f"Center 1 {self.center1} not in ligands"
        assert self.center2 in names, f"Center 2 {self.center2} not in ligands"
        edges = [(self.center1, self.center2)]
        for name in names:
            if name != self.center1 and name != self.center2:
                edges.append((self.center1, name))
                edges.append((self.center2, name))
        return edges


@NETWORK_GENERATOR_REGISTRY.register("custom")
class CustomNetworkGenerator(LigandNetworkGenerator):
    """Use a user-provided edge list directly."""

    def __init__(self, edges: list[tuple[str, str]]):
        self._edges = edges

    def run(self, ligands: list["Ligand"]) -> list[tuple[str, str]]:
        names = _check_unique_names(ligands)
        _check_edges(names, self._edges)
        return self._edges


try:
    from .openfe_impl import (
        OpenFELomapNetworkGenerator,
        OpenFEMinimalRedundantNetworkGenerator,
        OpenFEMinimalSpanningNetworkGenerator,
    )
except ImportError:
    pass


def load_network_generator(name: str, raise_errors: bool = False, **kwargs: Any) -> LigandNetworkGenerator:
    """
    Instantiate a ligand network generator from registry by name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``'star'``, ``'lomap'``, ``'minimal_spanning'``).
    raise_errors : bool, optional
        Reserved for API symmetry with other loaders. Exceptions are raised in
        all cases.
    **kwargs
        Passed to generator constructor.
    """
    del raise_errors
    if name not in NETWORK_GENERATOR_REGISTRY.names():
        available = ", ".join(NETWORK_GENERATOR_REGISTRY.names())
        raise NotImplementedError(f"Network generator {name!r} is not available. Available: {available}.")
    return NETWORK_GENERATOR_REGISTRY.create(name, **kwargs)
        