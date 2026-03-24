"""
OpenFE-backed ligand network generators.

These generators wrap OpenFE ligand-network planners and convert the resulting
OpenFE `LigandNetwork` into easybfe edge tuples of ligand names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TYPE_CHECKING

import openfe

from . import LigandNetworkGenerator, _check_edges, _check_unique_names
from .registry import NETWORK_GENERATOR_REGISTRY

if TYPE_CHECKING:
    from ..core import Ligand


def _to_openfe_components(ligands: list["Ligand"]) -> list[Any]:
    """Convert easybfe ligands to OpenFE SmallMoleculeComponents."""
    components: list[Any] = []
    for lig in ligands:
        rdmol = lig.get_rdmol()
        component = openfe.SmallMoleculeComponent.from_rdkit(rdmol, name=lig.name)
        components.append(component)
    return components


def _mapping_to_name_edge(mapping: Any) -> tuple[str, str]:
    """Extract `(nameA, nameB)` from an OpenFE ligand atom mapping object."""
    if hasattr(mapping, "componentA") and hasattr(mapping, "componentB"):
        return (mapping.componentA.name, mapping.componentB.name)
    if hasattr(mapping, "molA") and hasattr(mapping, "molB"):
        return (mapping.molA.name, mapping.molB.name)
    if hasattr(mapping, "component_a") and hasattr(mapping, "component_b"):
        return (mapping.component_a.name, mapping.component_b.name)
    raise AttributeError("Unable to extract component names from OpenFE mapping edge.")


def _network_to_edges(network: Any) -> list[tuple[str, str]]:
    """Convert OpenFE LigandNetwork edges to easybfe edge tuples."""
    return [_mapping_to_name_edge(edge) for edge in network.edges]


class _BaseOpenFENetworkGenerator(LigandNetworkGenerator):
    """Common OpenFE planner wrapper with mapper/scorer defaults."""

    def __init__(
        self,
        *,
        mappers: Any | Iterable[Any] | None = None,
        scorer: Callable[[Any], float] | None = None,
    ):
        if mappers is None:
            mappers = openfe.setup.atom_mapping.LomapAtomMapper()
        if scorer is None:
            scorer = openfe.setup.atom_mapping.lomap_scorers.default_lomap_score
        self.mappers = mappers
        self.scorer = scorer

    def _run_planner(self, ligands: list["Ligand"], planner: Callable[..., Any], **kwargs: Any) -> list[tuple[str, str]]:
        names = _check_unique_names(ligands)
        components = _to_openfe_components(ligands)
        network = planner(components, self.mappers, self.scorer, **kwargs)
        edges = _network_to_edges(network)
        _check_edges(names, edges)
        return edges


@NETWORK_GENERATOR_REGISTRY.register("minimal_spanning")
class OpenFEMinimalSpanningNetworkGenerator(_BaseOpenFENetworkGenerator):
    """Generate network with maximum-score minimum edge count."""

    def __init__(
        self,
        *,
        mappers: Any | Iterable[Any] | None = None,
        scorer: Callable[[Any], float] | None = None,
        progress: bool = True,
        n_processes: int = 1,
    ):
        super().__init__(mappers=mappers, scorer=scorer)
        self.progress = progress
        self.n_processes = n_processes

    def run(self, ligands: list["Ligand"]) -> list[tuple[str, str]]:
        return self._run_planner(
            ligands,
            openfe.setup.ligand_network_planning.generate_minimal_spanning_network,
            progress=self.progress,
            n_processes=self.n_processes,
        )


@NETWORK_GENERATOR_REGISTRY.register("lomap")
class OpenFELomapNetworkGenerator(_BaseOpenFENetworkGenerator):
    """Generate network using Lomap network construction rules."""

    def __init__(
        self,
        *,
        mappers: Any | Iterable[Any] | None = None,
        scorer: Callable[[Any], float] | None = None,
        distance_cutoff: float = 0.4,
        max_path_length: int = 6,
        actives: list[bool] | None = None,
        max_dist_from_active: int = 2,
        require_cycle_covering: bool = True,
        radial: bool = False,
        fast: bool = False,
        hub: Any | None = None,
    ):
        super().__init__(mappers=mappers, scorer=scorer)
        self.distance_cutoff = distance_cutoff
        self.max_path_length = max_path_length
        self.actives = actives
        self.max_dist_from_active = max_dist_from_active
        self.require_cycle_covering = require_cycle_covering
        self.radial = radial
        self.fast = fast
        self.hub = hub

    def run(self, ligands: list["Ligand"]) -> list[tuple[str, str]]:
        names = _check_unique_names(ligands)
        components = _to_openfe_components(ligands)
        hub = self.hub
        if isinstance(hub, str):
            by_name = {comp.name: comp for comp in components}
            assert hub in by_name, f"{hub} not in ligands"
            hub = by_name[hub]

        network = openfe.setup.ligand_network_planning.generate_lomap_network(
            components,
            self.mappers,
            self.scorer,
            distance_cutoff=self.distance_cutoff,
            max_path_length=self.max_path_length,
            actives=self.actives,
            max_dist_from_active=self.max_dist_from_active,
            require_cycle_covering=self.require_cycle_covering,
            radial=self.radial,
            fast=self.fast,
            hub=hub,
        )
        edges = _network_to_edges(network)
        _check_edges(names, edges)
        return edges


@NETWORK_GENERATOR_REGISTRY.register("minimal_redundant")
class OpenFEMinimalRedundantNetworkGenerator(_BaseOpenFENetworkGenerator):
    """Generate network with configurable MST redundancy."""

    def __init__(
        self,
        *,
        mappers: Any | Iterable[Any] | None = None,
        scorer: Callable[[Any], float] | None = None,
        progress: bool = True,
        mst_num: int = 2,
        n_processes: int = 1,
    ):
        super().__init__(mappers=mappers, scorer=scorer)
        self.progress = progress
        self.mst_num = mst_num
        self.n_processes = n_processes

    def run(self, ligands: list["Ligand"]) -> list[tuple[str, str]]:
        return self._run_planner(
            ligands,
            openfe.setup.ligand_network_planning.generate_minimal_redundant_network,
            progress=self.progress,
            mst_num=self.mst_num,
            n_processes=self.n_processes,
        )
