import numpy as np
from scipy.spatial.distance import cdist
import openmm.app as app
from rdkit import Chem

try:
    from .interaction_plip import *
except:
    pass


class HBondFinder:

    HBOND_THRESH = 3.2

    def __init__(self, protein_top: app.Topology, ligand_mol: Chem.Mol):
        self.hbond_acc_protein = []
        self.hbond_acc_ligand = []
        self.hbond_don_protein = []
        self.hbond_don_ligand = []

        # Protein
        for residue in protein_top.residues():
            for atom in residue.atoms():
                # backbone
                if atom.name == 'O':
                    self.hbond_acc_protein.append(atom.index)
                elif atom.name == 'N':
                    self.hbond_don_protein.append(atom.index)
                # side chains
                elif (residue.name in ('ARG', 'LYS', 'TRP')) and atom.name.startswith('N'):
                    self.hbond_don_protein.append(atom.index)
                elif (residue.name in ('SER', 'THR', 'TYR')) and atom.name.startswith('O'):
                    self.hbond_don_protein.append(atom.index)
                    self.hbond_acc_protein.append(atom.index)
                elif residue.name == 'ASN' or residue.name == 'GLN':
                    if atom.name.startswith('O'):
                        self.hbond_acc_protein.append(atom.index)
                    elif atom.name.startswith('N'):
                        self.hbond_don_protein.append(atom.index)
                elif residue.name == 'GLU' or residue.name == 'ASP':
                    if atom.name.startswith('O'):
                        self.hbond_acc_protein.append(atom.index)
                        bond_to_hydrogen = False
                        for bo in residue.bonds():
                            if bo.atom1 is atom and bo.atom2.name.startswith('H'):
                                bond_to_hydrogen = True
                                break
                            elif bo.atom2 is atom and bo.atom1.name.startswith('H'):
                                bond_to_hydrogen = True
                                break
                        if bond_to_hydrogen:
                            self.hbond_don_protein.append(atom.index)
                elif residue.name in ('HIS', 'HID', 'HIE', 'HIP'):
                    if atom.name.startswith('N'):
                        bond_to_hydrogen = False
                        for bo in residue.bonds():
                            if bo.atom1 is atom and bo.atom2.name.startswith('H'):
                                bond_to_hydrogen = True
                                break
                            elif bo.atom2 is atom and bo.atom1.name.startswith('H'):
                                bond_to_hydrogen = True
                                break
                        if bond_to_hydrogen:
                            self.hbond_don_protein.append(atom.index)
                        else:
                            self.hbond_acc_protein.append(atom.index)
        
        # Ligand
        for atom in ligand_mol.GetAtoms():
            if atom.GetSymbol() == 'O':
                if atom.GetFormalCharge() <= 0:
                    self.hbond_acc_ligand.append(atom.GetIdx())
                if any([nei.GetSymbol() == 'H' for nei in atom.GetNeighbors()]):
                    self.hbond_don_ligand.append(atom.GetIdx())
            elif atom.GetSymbol() == 'F':
                self.hbond_acc_ligand.append(atom.GetIdx())
            elif atom.GetSymbol() == 'N':
                if any([nei.GetSymbol() == 'H' for nei in atom.GetNeighbors()]):
                    self.hbond_don_ligand.append(atom.GetIdx())
                    # TODO: primary/secondary amine, imines with H should also be an acceptor
                    # although they should be protonated 
                else:
                    self.hbond_acc_ligand.append(atom.GetIdx())
    
    def apply(self, protein_pos: np.ndarray, ligand_pos: np.ndarray):
        hbond_data = []
        dist1 = cdist(protein_pos[self.hbond_acc_protein], ligand_pos[self.hbond_don_ligand])
        for row in np.argwhere(dist1 < self.HBOND_THRESH):
            pidx = int(self.hbond_acc_protein[row[0]])
            lidx = int(self.hbond_don_ligand[row[1]])
            hbond_data.append((pidx, lidx, dist1[row[0], row[1]], False))
        dist2 = cdist(protein_pos[self.hbond_don_protein], ligand_pos[self.hbond_acc_ligand])
        for row in np.argwhere(dist2 < self.HBOND_THRESH):
            pidx = int(self.hbond_don_protein[row[0]])
            lidx = int(self.hbond_acc_ligand[row[1]])
            hbond_data.append((pidx, lidx, dist2[row[0], row[1]], True))
        return hbond_data


class CloseContactFinder:
    """Find close contacts between protein and ligand heavy atoms."""

    DIST_THRESH = 3.5

    def __init__(self, protein_top: app.Topology, ligand_mol: Chem.Mol):
        self.heavy_atom_protein = []
        self.heavy_atom_ligand = []

        # Protein: heavy atoms only (exclude hydrogen)
        for residue in protein_top.residues():
            for atom in residue.atoms():
                if atom.element.symbol != "H":
                    self.heavy_atom_protein.append(atom.index)

        # Ligand: heavy atoms only (exclude hydrogen)
        for atom in ligand_mol.GetAtoms():
            if atom.GetSymbol() != "H":
                self.heavy_atom_ligand.append(atom.GetIdx())

    def find(self, protein_pos: np.ndarray, ligand_pos: np.ndarray):
        """Return list of (protein_atom_idx, ligand_atom_idx, distance) for pairs within DIST_THRESH."""
        dist = cdist(
            protein_pos[self.heavy_atom_protein], ligand_pos[self.heavy_atom_ligand]
        )
        result = []
        for row in np.argwhere(dist < self.DIST_THRESH):
            pidx = int(self.heavy_atom_protein[row[0]])
            lidx = int(self.heavy_atom_ligand[row[1]])
            d = float(dist[row[0], row[1]])
            result.append((pidx, lidx, d))
        return result