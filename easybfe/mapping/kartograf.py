from kartograf import KartografAtomMapper as _KartografAtomMapper
from ..core import Ligand
from .base import LigandRbfeAtomMapper
from .registry import MAPPER_REGISTRY


@MAPPER_REGISTRY.register("kartograf")
class KartografAtomMapper(LigandRbfeAtomMapper):

    def __init__(
        self, 
        *,
        atom_max_distance: float = 0.95,
        atom_map_hydrogens: bool = True,
        map_hydrogens_on_hydrogens_only: bool = False,
        map_exact_ring_matches_only: bool = True,
        allow_partial_fused_rings: bool = True,
        allow_bond_breaks: bool = False,
        **kwargs
    ):
        """ Wrapper of Kartograf - a geometry-based atom mapper

        Parameters
        ----------
        atom_max_distance : float
            geometric criteria for two atoms, how far their distance
            can be maximal (in Angstrom). Default 0.95
        map_hydrogens_on_hydrogens_only : bool
            map hydrogens only on hydrogens. Default False
        map_exact_ring_matches_only : bool
            if true, only rings with matching ringsize and same bond-orders
            will be mapped. Additionally no ring-breaking is permitted. default
            False
        allow_bond_breaks : bool
            if False, automatically applies ``filter_bond_breaks`` to avoid
            mappings where bonds are broken. default False
        allow_partial_fused_rings: bool
            If we should allow partially mapped fused rings (True) or not (False). Default True.

        **kwargs
            Passed to :class:`~easybfe.mapping.base.LigandRbfeAtomMapper` (e.g.
            ``allow_map_hydrogen_to_non_hydrogen``, ``allow_hybridization_change``).

        """
        super().__init__(**kwargs)
        self._mapper = _KartografAtomMapper(
            atom_max_distance=atom_max_distance,
            atom_map_hydrogens=atom_map_hydrogens,
            map_hydrogens_on_hydrogens_only=map_hydrogens_on_hydrogens_only,
            map_exact_ring_matches_only=map_exact_ring_matches_only,
            allow_partial_fused_rings=allow_partial_fused_rings,
            allow_bond_breaks=allow_bond_breaks
        )

    def propose_mapping(self, ligandA: Ligand, ligandB: Ligand):
        return self._mapper.suggest_mapping_from_rdmols(ligandA.get_rdmol(), ligandB.get_rdmol())