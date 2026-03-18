from lomap.mcs import MCS as LomapMCS
from ..core import Ligand
from .base import LigandRbfeAtomMapper
from .registry import MAPPER_REGISTRY


@MAPPER_REGISTRY.register("lomap")
class LomapAtomMapper(LigandRbfeAtomMapper):

    def __init__(self, *, time: int = 20, threed: bool = True,
                 max3d: float = 1.0, element_change: bool = True,
                 seed: str = '', shift: bool = False, **kwargs):
        """Wraps the MCS atom mapper from Lomap.

        Kwargs are passed directly to the MCS class from Lomap for each mapping
        created. Additional kwargs are forwarded to :class:`LigandRbfeAtomMapper`.

        Parameters
        ----------
        time : int, optional
          timeout of MCS algorithm, passed to RDKit
          default 20
        threed : bool, optional
          if true, positional info is used to choose between symmetrically
          equivalent mappings and prune the mapping, default True
        max3d : float, optional
          maximum discrepancy in Angstroms between atoms before mapping is not
          allowed, default 1000.0, which effectively trims no atoms
        element_change: bool, optional
          whether to allow element changes in the mappings, default True
        seed: str, optional
          SMARTS string to use as seed for MCS searches.  When used across an
          entire set of ligands, this can speed up calculations considerably
        shift: bool, optional
          when determining 3D overlap, if to translate the two molecules MCS to minimise
          RMSD to boost potential alignment.
        **kwargs
          Passed to :class:`~easybfe.mapping.base.LigandRbfeAtomMapper` (e.g.
          ``allow_element_change``, ``allow_map_hydrogen_to_non_hydrogen``,
          ``allow_hybridization_change``).
        """
        element_change = kwargs.pop("allow_element_change", element_change)
        super().__init__(allow_element_change=element_change, **kwargs)
        self.time = time
        self.threed = threed
        self.max3d = max3d
        self.element_change = element_change
        self.seed = seed
        self.shift = shift

    def propose_mapping(self, ligandA: Ligand, ligandB: Ligand):
        try:
            mcs = LomapMCS(ligandA.get_rdmol(), ligandB.get_rdmol(),
                            time=self.time,
                            threed=self.threed, max3d=self.max3d,
                            element_change=self.element_change,
                            seed=self.seed,
                            shift=self.shift)
        except ValueError:
            # if no match found, Lomap throws ValueError, so we just yield
            # generator with no contents
            return {}

        mapping_string = mcs.all_atom_match_list()
        # lomap spits out "1:1,2:2,...,x:y", so split around commas,
        # then colons and coerce to ints
        mapping_dict = dict((map(int, v.split(':'))
                             for v in mapping_string.split(',')))
        return mapping_dict