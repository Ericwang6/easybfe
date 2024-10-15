import os
import pytest

from easybfe.amber.project import AmberRbfeProject
from easybfe.amber.prep import prep_ligand_rbfe_systems



class TestAmberLigandRbfeProject:
    @classmethod
    def setup_class(cls):
        cls.project = AmberRbfeProject(os.path.join(os.path.dirname(__file__), '_test_ligand_rbfe_project'))
    
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
        self.project.add_ligand(
            fpath=os.path.join(os.path.dirname(__file__), 'data/CDD_1845.sdf'),
            name='CDD_1845',
            protein_name='1845',
            parametrize=True,
            forcefield='gaff2',
            charge_method='gas',
            overwrite=True
        )
        assert os.path.isfile(self.project.ligands_dir / '1845/CDD_1845/CDD_1845.sdf')
        assert os.path.isfile(self.project.ligands_dir / '1845/CDD_1845/gaff2_gas/CDD_1845.prmtop')
        assert os.path.isfile(self.project.ligands_dir / '1845/CDD_1845/gaff2_gas/CDD_1845.top')
    
    @pytest.mark.dependency(name='test_add_ligand_custom', dependency=['test_add_protein'])
    def test_add_ligand_custom(self):
        self.project.add_ligand(
            fpath=os.path.join(os.path.dirname(__file__), 'data/CDD_1819.sdf'),
            name='CDD_1819',
            protein_name='1845',
            parametrize=True,
            custom_ff=os.path.join(os.path.dirname(__file__), 'data/CDD_1819.prmtop'),
            overwrite=True
        )
        assert os.path.isfile(self.project.ligands_dir / '1845/CDD_1819/CDD_1819.sdf')
        assert os.path.isfile(self.project.ligands_dir / '1845/CDD_1819/custom/CDD_1819.prmtop')
    
    @pytest.mark.dependency(name='test_add_ligand_openff', dependency=['test_add_protein'])
    def test_add_ligand_openff(self):
        self.project.parametrize_ligand(
            name='CDD_1819',
            protein_name='1845',
            forcefield='openff-2.1.0',
            charge_method='gas',
            overwrite=True
        )
        assert os.path.isfile(self.project.ligands_dir / '1845/CDD_1819/CDD_1819.sdf')
        assert os.path.isfile(self.project.ligands_dir / '1845/CDD_1819/openff-2.1.0_gas/CDD_1819.prmtop')
    
    @pytest.mark.dependency(name='test_perturbation', dependency=['test_add_protein', 'test_add_ligand', 'test_add_ligand_custom'])
    def test_add_perturbation(self):
        self.project.add_perturbation(
            ligandA_name='CDD_1819',
            ligandB_name='CDD_1845',
            protein_name='1845',
            ligandA_ff='custom/prmtop',
            ligandB_ff='gaff2_gas/prmtop',
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