"""Single-structure Boresch restraint finders.

Contains the geometry-based :class:`RxRxBoreschRestraintsFinder` (H-bonds with a
close-contact fallback) and :class:`UserSpecifiedBoreschRestraint` (explicit
anchors).
"""

import logging
from typing import Optional

import numpy as np

from ..core import Protein, Ligand
from ..analysis.interaction import HBondFinder, CloseContactFinder
from .base import BoreschRestraintsFinder, BORESCH_FINDER_REGISTRY
from .restraint import BoreschRestraint
from .utils import compute_angle


logger = logging.getLogger(__name__)


@BORESCH_FINDER_REGISTRY.register("rxrx")
class RxRxBoreschRestraintsFinder(BoreschRestraintsFinder):
    """Boresch restraint finder that selects anchors via H-bonds and close contacts (RxRx-style)."""

    def __init__(
        self,
        protein: Protein,
        ligand: Ligand,
        wts: tuple[float, float, float, float, float, float] = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        *args,
        **kwargs,
    ):
        super().__init__(protein, ligand, wts, *args, **kwargs)

    def find(
        self,
        protein_positions: Optional[np.ndarray] = None,
        ligand_positions: Optional[np.ndarray] = None,
    ) -> BoreschRestraint:
        protein_pdb = self.protein.to_openmm()
        if protein_positions is None:
            protein_positions = np.array([[v.x, v.y, v.z] for v in protein_pdb.positions]) * 10
        if ligand_positions is None:
            ligand_mol = self.ligand.get_rdmol()
            ligand_positions = ligand_mol.GetConformer().GetPositions()
        protein_pos = protein_positions
        ligand_pos = ligand_positions
        protein_atoms = list(protein_pdb.topology.atoms())
        ligand_mol = self.ligand.get_rdmol()
        hbond_finder = HBondFinder(protein_pdb.topology, ligand_mol)
        hbond_data = hbond_finder.apply(protein_pos, ligand_pos)
        logger.info("Found %d hydrogen bonds between protein and ligand", len(hbond_data))

        def _find_candidates(pairs):
            candidates = []
            for pidx, lidx in pairs:
                ligand_atom = ligand_mol.GetAtomWithIdx(lidx)

                if sum([nei.GetSymbol() != 'H' for nei in ligand_atom.GetNeighbors()]) < 2:
                    continue

                protein_atom = protein_atoms[pidx]
                residue = protein_atom.residue
                bb_ca_atoms = [at for at in residue.atoms() if at.name == 'CA']
                bb_c_atoms = [at for at in residue.atoms() if at.name == 'C']
                if not bb_ca_atoms or not bb_c_atoms:
                    continue
                bb_ca = bb_ca_atoms[0].index
                bb_c = bb_c_atoms[0].index
                bb_n = None
                for bo in residue.bonds():  
                    if bo.atom1 is protein_atoms[bb_c] and bo.atom2.name == 'N':
                        bb_n = bo.atom2.index
                    elif bo.atom2 is protein_atoms[bb_c] and bo.atom1.name == 'N':
                        bb_n = bo.atom1.index
                if bb_n is None:
                    continue

                for nei in ligand_atom.GetNeighbors():
                    if nei.GetSymbol() == 'H':
                        continue
                    ang1 = compute_angle(protein_pos[bb_c], protein_pos[bb_ca], ligand_pos[lidx])
                    ang2 = compute_angle(protein_pos[pidx], ligand_pos[lidx], ligand_pos[nei.GetIdx()])
                    if ang1 < 30 or ang1 > 150 or ang2 < 30 or ang2 > 150:
                        continue
                    for nnei in nei.GetNeighbors():
                        if nei.GetSymbol() != 'H' and nnei.GetSymbol() != 'H' and nnei.GetIdx() != lidx:
                            candidates.append((bb_ca, bb_c, bb_n, lidx, nei.GetIdx(), nnei.GetIdx()))
            return candidates

        candidates = _find_candidates([(r[0], r[1]) for r in hbond_data])
        logger.info("Found %d Boresch candidates from hydrogen bonds", len(candidates))
        if len(candidates) == 0:
            close_contact_finder = CloseContactFinder(protein_pdb.topology, ligand_mol)
            close_contacts = close_contact_finder.find(protein_pos, ligand_pos)
            logger.info("Found %d close contacts", len(close_contacts))
            candidates = _find_candidates([(r[0], r[1]) for r in close_contacts])
            logger.info("Found %d Boresch candidates from close contacts", len(candidates))

        if len(candidates) == 0:
            raise ValueError(
                "Could not find suitable Boresch restraint anchor candidates "
                "from either hydrogen bonds or close contacts. This means the ligand is positioned in a weird pose"
            )
            
        
        # Compute ligand center of mass from RDKit geometry (in Angstroms)
        masses = np.array([atom.GetMass() for atom in ligand_mol.GetAtoms()])
        ligand_com = np.average(ligand_pos, axis=0, weights=masses)

        # Sort candidates by distance of primary ligand anchor (L1) to COM
        candidates.sort(key=lambda x: np.linalg.norm(ligand_pos[x[3]] - ligand_com))

        restr = BoreschRestraint(
            protein_anchors=tuple(candidates[0][:3]),
            ligand_anchors=tuple(candidates[0][3:]),
            rst_wts=self.wts,
        )
        restr.compute_rst_vals(protein_pos, ligand_pos)
        return restr


@BORESCH_FINDER_REGISTRY.register("user")
class UserSpecifiedBoreschRestraint(BoreschRestraintsFinder):
    """Boresch restraint finder that uses user-provided protein and ligand anchor indices."""

    def __init__(
        self,
        protein: Protein,
        ligand: Ligand,
        protein_anchors: tuple[int, int, int],
        ligand_anchors: tuple[int, int, int],
        wts: tuple[float, float, float, float, float, float] = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        *args,
        **kwargs,
    ):
        super().__init__(protein, ligand, wts, *args, **kwargs)
        self.protein_anchors = protein_anchors
        self.ligand_anchors = ligand_anchors

    def find(
        self,
        protein_positions: Optional[np.ndarray] = None,
        ligand_positions: Optional[np.ndarray] = None,
    ) -> BoreschRestraint:
        restr = BoreschRestraint(
            protein_anchors=self.protein_anchors,
            ligand_anchors=self.ligand_anchors,
            rst_wts=self.wts,
        )
        if protein_positions is None:
            protein_pdb = self.protein.to_openmm()
            protein_positions = np.array([[v.x, v.y, v.z] for v in protein_pdb.positions]) * 10
        if ligand_positions is None:
            ligand_positions = self.ligand.get_rdmol().GetConformer().GetPositions()
        restr.compute_rst_vals(protein_positions, ligand_positions)
        return restr
