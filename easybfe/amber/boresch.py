import math
import numpy as np
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type

from ..config import AmberRstSettings
from ..core import Protein, Ligand
from ..analysis.interaction import HBondFinder, CloseContactFinder


logger = logging.getLogger(__name__)


def compute_bond(pos0, pos1):
    dx, dy, dz = pos1[0]-pos0[0], pos1[1]-pos0[1], pos1[2]-pos0[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def compute_angle(pos0, pos1, pos2):
    """
    Compute the angle between three positions.
    
    Computes the angle at pos1 formed by the vectors pos0->pos1 and pos1->pos2.
    
    Parameters
    ----------
    pos0 : array-like
        First position (x, y, z).
    pos1 : array-like
        Central position (x, y, z).
    pos2 : array-like
        Third position (x, y, z).
    
    Returns
    -------
    float
        Angle in degrees, in the range (0, 180).
    """
    # Convert to numpy arrays for vector operations
    pos0 = np.array(pos0)
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    
    # Compute vectors
    vec1 = pos0 - pos1  # Vector from pos1 to pos0
    vec2 = pos2 - pos1  # Vector from pos1 to pos2
    
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute angle: zero-length vector")
    
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2
    
    # Compute angle using dot product
    cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def compute_dihedral(pos0, pos1, pos2, pos3):
    """
    Compute the dihedral angle (torsion angle) between four positions.
    
    Computes the dihedral angle defined by the four positions pos0-pos1-pos2-pos3.
    This is the angle between the planes defined by pos0-pos1-pos2 and pos1-pos2-pos3.
    
    Parameters
    ----------
    pos0 : array-like
        First position (x, y, z).
    pos1 : array-like
        Second position (x, y, z).
    pos2 : array-like
        Third position (x, y, z).
    pos3 : array-like
        Fourth position (x, y, z).
    
    Returns
    -------
    float
        Dihedral angle in degrees, in the range (-180, 180).
    """
    # Convert to numpy arrays for vector operations
    pos0 = np.array(pos0)
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    pos3 = np.array(pos3)
    
    # Compute vectors
    vec1 = pos1 - pos0  # Vector from pos0 to pos1
    vec2 = pos2 - pos1  # Vector from pos1 to pos2
    vec3 = pos3 - pos2  # Vector from pos2 to pos3
    
    # Compute cross products to get normal vectors to the planes
    cross1 = np.cross(vec1, vec2)  # Normal to plane pos0-pos1-pos2
    cross2 = np.cross(vec2, vec3)  # Normal to plane pos1-pos2-pos3
    
    # Normalize cross products
    norm1 = np.linalg.norm(cross1)
    norm2 = np.linalg.norm(cross2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute dihedral: degenerate geometry (atoms are collinear)")
    
    cross1 = cross1 / norm1
    cross2 = cross2 / norm2
    
    # Normalize vec2 for sign determination
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec2 == 0:
        raise ValueError("Cannot compute dihedral: zero-length central bond")
    vec2_norm = vec2 / norm_vec2
    
    # Compute sin and cos of dihedral angle
    cos_angle = np.clip(np.dot(cross1, cross2), -1.0, 1.0)
    sin_angle = np.dot(np.cross(cross1, cross2), vec2_norm)
    
    # Use atan2 for proper sign determination (returns angle in [-pi, pi])
    dihedral_rad = math.atan2(sin_angle, cos_angle)
    dihedral_deg = math.degrees(dihedral_rad)
    
    return dihedral_deg


@dataclass
class BoreschRestraint:
    protein_anchors: tuple
    ligand_anchors: tuple
    rst_wts: tuple = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    rst_vals: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def compute_rst_vals(self, protein_positions: np.ndarray, ligand_positions: np.ndarray):
        """
        Compute Boresch restraint values from protein and ligand positions.
        
        Computes the six geometric parameters (r0, alpha0, theta0, gamma0, beta0, phi0)
        that define the Boresch restraints based on the current positions of the anchor atoms.
        
        Parameters
        ----------
        protein_positions : np.ndarray
            Array of protein atom positions, shape (n_protein_atoms, 3).
        ligand_positions : np.ndarray
            Array of ligand atom positions, shape (n_ligand_atoms, 3).
        
        Returns
        -------
        tuple[float, float, float, float, float, float]
            Tuple containing (r0, alpha0, theta0, gamma0, beta0, phi0) in degrees for angles/dihedrals.
            r0 is in Angstroms.
        """
        l1, l2, l3 = self.ligand_anchors
        p1, p2, p3 = self.protein_anchors
        
        # Extract positions for anchor atoms
        pos_l1 = ligand_positions[l1]
        pos_l2 = ligand_positions[l2]
        pos_l3 = ligand_positions[l3]
        pos_p1 = protein_positions[p1]
        pos_p2 = protein_positions[p2]
        pos_p3 = protein_positions[p3]
        
        # Compute distance: L1-P1
        r0 = compute_bond(pos_l1, pos_p1)
        
        # Compute angle: P1-L1-L2
        alpha0 = compute_angle(pos_p1, pos_l1, pos_l2)
        
        # Compute angle: P2-P1-L1
        theta0 = compute_angle(pos_p2, pos_p1, pos_l1)
        
        # Compute dihedral: P1-L1-L2-L3
        gamma0 = compute_dihedral(pos_p1, pos_l1, pos_l2, pos_l3)
        
        # Compute dihedral: P2-P1-L1-L2
        beta0 = compute_dihedral(pos_p2, pos_p1, pos_l1, pos_l2)
        
        # Compute dihedral: P3-P2-P1-L1
        phi0 = compute_dihedral(pos_p3, pos_p2, pos_p1, pos_l1)
        
        vals = (r0, alpha0, theta0, gamma0, beta0, phi0)
        self.rst_vals = vals
        return vals

    def make_rst(self, offset: int):
        """Construct multiple Boresch restraints.
        
        Parameters
        ----------
        offset : int
            Offset to add to protein atom indices (typically number of ligand atoms).
        
        Returns
        -------
        list[AmberRstSettings]
            List of restraint settings for distance, angles, and dihedrals.
        """
        l1, l2, l3 = self.ligand_anchors
        p1, p2, p3 = self.protein_anchors
        r0, alpha0, theta0, gamma0, beta0, phi0 = self.rst_vals
        rk, alphak, thetak, gammak, betak, phik = self.rst_wts
        
        # Adjusted boundaries for dihedrals (accounts for periodicity)
        dih11, dih14 = gamma0 - 180.0, gamma0 + 180.0
        dih21, dih24 = beta0 - 180.0, beta0 + 180.0
        dih31, dih34 = phi0 - 180.0, phi0 + 180.0
        
        # Build the list of restraint settings
        # Note: iat uses 1-based indexing, so add 1 to all atom indices
        rst_list = [
            # Distance restraint: L1-P1
            AmberRstSettings(iat=[l1+1, p1+offset+1], r1=0.0, r2=r0, r3=r0, r4=99.0, rk2=rk, rk3=rk),
            # Angle restraint: P1-L1-L2
            AmberRstSettings(iat=[p1+offset+1, l1+1, l2+1], r1=0.0, r2=alpha0, r3=alpha0, r4=180.0, rk2=alphak, rk3=alphak),
            # Angle restraint: P2-P1-L1
            AmberRstSettings(iat=[p2+offset+1, p1+offset+1, l1+1], r1=0.0, r2=theta0, r3=theta0, r4=180.0, rk2=thetak, rk3=thetak),
            # Dihedral restraint: P1-L1-L2-L3
            AmberRstSettings(iat=[p1+offset+1, l1+1, l2+1, l3+1], r1=dih11, r2=gamma0, r3=gamma0, r4=dih14, rk2=gammak, rk3=gammak),
            # Dihedral restraint: P2-P1-L1-L2
            AmberRstSettings(iat=[p2+offset+1, p1+offset+1, l1+1, l2+1], r1=dih21, r2=beta0, r3=beta0, r4=dih24, rk2=betak, rk3=betak),
            # Dihedral restraint: P3-P2-P1-L1
            AmberRstSettings(iat=[p3+offset+1, p2+offset+1, p1+offset+1, l1+1], r1=dih31, r2=phi0, r3=phi0, r4=dih34, rk2=phik, rk3=phik)
        ]
        
        return rst_list
    

def compute_boresch_energy(rst_vals, rst_wts, temperature: float = 298.15):
    import numpy as np
    import scipy.special as special
    r0, alpha0, theta0, gamma0, beta0, phi0 = rst_vals
    rk, alphak, thetak, gammak, betak, phik = rst_wts

    V = 1660. # in A^3
    kBT = temperature * 0.0019872041 # Boltzmann's constant (kcal/mol/K)

    beta = 1 / kBT

    def _compute_Zr(r0, rk):
        return r0 / (2*beta*rk) * np.exp(-beta*rk*r0**2) + np.sqrt(np.pi) / (4*beta*rk*np.sqrt(beta*rk)) * (1+2*beta*r0**2*rk) * (1+special.erf(np.sqrt(beta*rk)*r0))
    
    def _compute_Ztheta(theta0, thetak):
        return np.sqrt(np.pi/(beta*thetak)) * np.exp(-1/(4*beta*thetak)) * np.sin(np.radians(theta0))
    
    def _compute_Zphi(phi, phik):
        return np.sqrt(np.pi/(beta*phik)) * special.erf(np.pi*np.sqrt(beta*phik))

    Zalpha = _compute_Ztheta(alpha0, alphak)
    Ztheta = _compute_Ztheta(theta0, thetak)
    Zr = _compute_Zr(r0, rk)
    Zgamma = _compute_Zphi(gamma0, gammak)
    Zbeta = _compute_Zphi(beta0, betak)
    Zphi = _compute_Zphi(phi0, phik)

    return -kBT*np.log(Zr*Zalpha*Ztheta*Zgamma*Zbeta*Zphi/(8*np.pi**2*V))


class BoreschRestraintsFinder(ABC):
    """Abstract base class for finding Boresch restraint anchors and building :class:`BoreschRestraint`."""

    def __init__(
        self,
        protein: Protein,
        ligand: Ligand,
        wts: tuple[float, float, float, float, float, float] = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        *args,
        **kwargs,
    ):
        self.protein = protein
        self.ligand = ligand
        self.wts = wts

    @abstractmethod
    def find(
        self,
        protein_positions: Optional[np.ndarray] = None,
        ligand_positions: Optional[np.ndarray] = None,
    ) -> BoreschRestraint:
        """Find anchor atoms and return a :class:`BoreschRestraint`.

        If ``protein_positions`` or ``ligand_positions`` are not provided, positions
        are taken from :attr:`protein` and :attr:`ligand` (e.g. from OpenMM/RDKit).
        """
        ...


# Registry for BoreschRestraintsFinder implementations
_BORESCH_FINDER_REGISTRY: dict[str, Type[BoreschRestraintsFinder]] = {}


def register_boresch_finder(name: str):
    """Register a :class:`BoreschRestraintsFinder` subclass under the given name.

    Use as a class decorator::

        @register_boresch_finder("RxRx")
        class RxRxBoreschRestraintsFinder(BoreschRestraintsFinder):
            ...
    """

    def decorator(cls: Type[BoreschRestraintsFinder]) -> Type[BoreschRestraintsFinder]:
        if not issubclass(cls, BoreschRestraintsFinder):
            raise TypeError(f"{cls.__name__} must be a subclass of BoreschRestraintsFinder")
        _BORESCH_FINDER_REGISTRY[name] = cls
        return cls

    return decorator


def get_boresch_finder(name: str) -> Type[BoreschRestraintsFinder]:
    """Return the :class:`BoreschRestraintsFinder` subclass registered under ``name``."""
    if name not in _BORESCH_FINDER_REGISTRY:
        available = ", ".join(sorted(_BORESCH_FINDER_REGISTRY))
        raise KeyError(f"Unknown Boresch finder {name!r}. Available: {available}")
    return _BORESCH_FINDER_REGISTRY[name]


def list_boresch_finders() -> list[str]:
    """Return the list of registered Boresch finder names."""
    return sorted(_BORESCH_FINDER_REGISTRY)


@register_boresch_finder("rxrx")
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


@register_boresch_finder("user")
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

