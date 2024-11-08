import os, shutil
import pytest
import json

import numpy as np
import parmed
from rdkit import Chem

from easybfe.amber.project import AmberRbfeProject
from easybfe.amber.prep import prep_ligand_rbfe_systems


class TestAmberLigandRbfeProject:
    @classmethod
    def setup_class(cls):
        proj_dir = os.path.join(os.path.dirname(__file__), '_test_ligand_rbfe_project')
        if os.path.isdir(proj_dir):
            shutil.rmtree(proj_dir)
        cls.project = AmberRbfeProject(proj_dir, init=True)
    
    @pytest.mark.dependency(name='test_add_protein')
    def test_add_protein(self):
        self.project.add_protein(
            os.path.join(os.path.dirname(__file__), 'data/CDD_1845.pdb'),
            name='1845',
            check_ff=True,
            overwrite=True
        )
        assert os.path.isfile(self.project.proteins_dir / '1845/1845.pdb')
    
    @pytest.mark.dependency(name='test_add_ligand', dependency=['test_add_protein'])
    def test_add_ligand(self):
        name = 'CDD_1845_gaff2'
        self.project.add_ligand(
            os.path.join(os.path.dirname(__file__), 'data/CDD_1845.sdf'),
            name=name,
            protein_name='1845',
            parametrize=True,
            forcefield='gaff2',
            charge_method='gas',
            overwrite=True
        )
        assert os.path.isfile(self.project.ligands_dir / f'1845/{name}/{name}.sdf')
        assert os.path.isfile(self.project.ligands_dir / f'1845/{name}/{name}.prmtop')
        assert os.path.isfile(self.project.ligands_dir / f'1845/{name}/{name}.top')
    
    @pytest.mark.dependency(name='test_add_ligand_custom', dependency=['test_add_protein'])
    def test_add_ligand_custom(self):
        name = 'CDD_1819_custom'
        self.project.add_ligand(
            os.path.join(os.path.dirname(__file__), 'data/CDD_1819.sdf'),
            name=name,
            protein_name='1845',
            parametrize=True,
            forcefield=os.path.join(os.path.dirname(__file__), 'data/CDD_1819.prmtop'),
            overwrite=True
        )
        assert os.path.isfile(self.project.ligands_dir / f'1845/{name}/{name}.sdf')
        assert os.path.isfile(self.project.ligands_dir / f'1845/{name}/{name}.prmtop')
    
    @pytest.mark.dependency(name='test_add_ligand_openff', dependency=['test_add_protein'])
    def test_add_ligand_openff(self):
        name = 'CDD_1819_off'
        self.project.add_ligand(
            os.path.join(os.path.dirname(__file__), 'data/CDD_1819.sdf'),
            name=name,
            protein_name='1845',
            forcefield='openff-2.1.0',
            charge_method='gas',
            overwrite=True
        )
        assert os.path.isfile(self.project.ligands_dir / f'1845/{name}/{name}.sdf')
        assert os.path.isfile(self.project.ligands_dir / f'1845/{name}/{name}.prmtop')
    
    @pytest.mark.dependency(name='test_perturbation', dependency=['test_add_protein', 'test_add_ligand', 'test_add_ligand_custom'])
    def test_add_perturbation(self):
        self.project.add_perturbation(
            ligandA_name='CDD_1819_custom',
            ligandB_name='CDD_1845_gaff2',
            protein_name='1845',
            pert_name='CDD_1819~CDD_1845',
            config=os.path.join(os.path.dirname(__file__), 'data/config_5ns.json'),
            overwrite=True
        )


def test_amber_ligand_rbfe_prep():
    pwd = os.path.dirname(__file__)
    datadir = os.path.join(pwd, 'data')
    wdir = os.path.join(pwd, '_test_prep')
    mapping = {
        0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 9, 8: 10, 9: 11, 
        10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19,
        18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 31, 25: 32,
        26: 33, 27: 34, 28: 35, 29: 36, 30: 37, 31: 38, 32: 39, 33: 40,
        34: 41, 35: 42, 36: 43, 37: 44, 38: 45, 39: 46, 40: 47, 41: 48, 
        42: 49, 43: 50, 44: 51, 45: 52, 46: 53, 47: 54, 48: 55, 49: 56, 
        50: 57, 51: 58, 52: 59, 53: 60, 54: 61
    }
    prep_ligand_rbfe_systems(
        protein_pdb=os.path.join(datadir, 'CDD_1845.pdb'),
        ligandA_mol=os.path.join(datadir, 'CDD_1819.sdf'),
        ligandA_top=os.path.join(datadir, 'CDD_1819.prmtop'),
        ligandB_mol=os.path.join(datadir, 'CDD_1845.sdf'),
        ligandB_top=os.path.join(datadir, 'CDD_1845.prmtop'),
        mapping=mapping,
        wdir=wdir
    )
    mask_ref = {
        "noshakemask": "'@1-145'",
        "timask1": "'@1-73'",
        "timask2": "'@74-145'",
        "scmask1": "'@56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73'",
        "scmask2": "'@129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145'"
    }
    with open(os.path.join(wdir, 'mask.json')) as f:
        mask = json.load(f)
        assert mask == mask_ref
    
    num_atoms_A = Chem.SDMolSupplier(os.path.join(datadir, 'CDD_1819.sdf'), removeHs=False)[0].GetNumAtoms()
    struct = parmed.load_file(
        os.path.join(wdir, 'gas.prmtop'),
        xyz=os.path.join(wdir, 'gas.inpcrd')
    )
    assert np.allclose(struct.coordinates[:len(mapping)], struct.coordinates[num_atoms_A:][:len(mapping)])


def test_amber_charge_change():
    datadir = os.path.join(os.path.dirname(__file__), 'data')
    proj_dir = os.path.join(os.path.dirname(__file__), '_test_ligand_rbfe_charge_change')
    if os.path.isdir(proj_dir):
        shutil.rmtree(proj_dir)
    proj = AmberRbfeProject(proj_dir, init=True)
    proj.add_protein(os.path.join(datadir, 'tyk2_pdbfixer.pdb'), name='tyk2')
    proj.add_ligand(os.path.join(datadir, 'jmc_23.sdf'), name='jmc_23', protein_name='tyk2', parametrize=True, charge_method='gas', overwrite=True)
    proj.add_ligand(os.path.join(datadir, 'jmc_32.sdf'), name='jmc_32', protein_name='tyk2', parametrize=True, charge_method='gas', overwrite=True)
    proj.add_perturbation('jmc_23', 'jmc_32', 'tyk2', config=os.path.join(datadir, 'config_5ns.json'))
    struct = parmed.load_file(os.path.join(proj_dir, 'rbfe/tyk2/jmc_23~jmc_32/prep/solvent.prmtop'))
    for residue in struct.residues:
        if residue.name == 'ALW':
            assert np.allclose([at.mass for at in residue.atoms[-2:]], 3.024)
    assert struct.angles[-1].atom1.residue.name == 'ALW'