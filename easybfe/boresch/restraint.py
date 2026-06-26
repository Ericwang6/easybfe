"""The :class:`BoreschRestraint` data container and free-energy correction.

Holds the six Boresch anchor atoms and degrees of freedom, converts them into
AMBER restraint settings, and computes the analytical Boresch free-energy
correction.
"""

from dataclasses import dataclass

import numpy as np

from ..config import AmberRstSettings
from .utils import compute_bond, compute_angle, compute_dihedral


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
