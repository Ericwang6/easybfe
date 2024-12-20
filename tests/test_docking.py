from pathlib import Path
import pytest
from rdkit import Chem
from easybfe.docking import compute_box_from_coordinates, VinaDocking


def test_vina():
    datadir = Path(__file__).parent / 'data'
    ligand_sdf = datadir / 'CDD_1819.sdf'
    protein_pdb = datadir / 'CDD_1845.pdb'
    mol = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)[0]
    center, box = compute_box_from_coordinates(mol.GetConformer().GetPositions())
    vina = VinaDocking(protein=protein_pdb, box_center=center, box_size=box, wdir=datadir.parent / '_test_vina', protein_prep_tool='obabel')
    vina.dock(ligand_sdf)


def test_vina_constr():
    datadir = Path(__file__).parent / 'data'
    ligand_sdf = datadir / 'CDD_1819.sdf'
    ref_sdf = datadir / 'CDD_1845.sdf'
    protein_pdb = datadir / 'CDD_1845.pdb'
    mol = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)[0]
    center, box = compute_box_from_coordinates(mol.GetConformer().GetPositions())
    vina = VinaDocking(protein=protein_pdb, box_center=center, box_size=box, wdir=datadir.parent / '_test_vina_constr', protein_prep_tool='obabel')
    vina.constr_dock(ligand_sdf, ref_sdf)