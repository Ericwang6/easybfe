import shutil
from pathlib import Path
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from easybfe.smff import load_parametrizer
from easybfe.core.ligand import LigandLoader


@pytest.fixture
def testdir():
    """Create and clean up test directory."""
    testdir = Path(__file__).parent / '_test_smff'
    if testdir.is_dir():
        shutil.rmtree(testdir)
    testdir.mkdir()
    yield testdir


@pytest.fixture
def amide_sdf():
    """Path to amide test file."""
    return Path(__file__).parent / 'data' / 'amide.sdf'


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ('gaff', 'bcc'),
        ('gaff', 'gas'),
        ('gaff2', 'bcc'),
        ('gaff2', 'gas'),
        ('openff-2.1.0', 'bcc'),
        ('openff-2.1.0', 'gas'),
    ]
)
def test_parametrize_ligand_with_file(forcefield, charge_method, testdir, amide_sdf):
    """Test parametrization with file input using LigandLoader and run method."""
    wdir = testdir / f'{forcefield}_{charge_method}_file'
    
    # Load ligand from file using LigandLoader
    loader = LigandLoader()
    ligands = loader.load(amide_sdf, only_first=True, name_from_stem=True)
    assert len(ligands) == 1
    ligand = ligands[0]
    
    # Run parametrization
    ff = load_parametrizer(forcefield, charge_method)
    ligand = ff.run(ligand)
    
    # Check that auxiliary files are present
    assert 'prmtop' in ligand.auxiliary_files
    assert 'inpcrd' in ligand.auxiliary_files
    assert 'xml' in ligand.auxiliary_files
    assert 'pdb' in ligand.auxiliary_files
    
    # Write files to directory for checking
    ligand.dump(wdir)
    
    # Check that required files are generated
    stem = amide_sdf.stem
    assert (wdir / f'{stem}.prmtop').is_file()
    assert (wdir / f'{stem}.inpcrd').is_file()
    assert (wdir / f'{stem}.sdf').is_file()
    assert (wdir / f'{stem}.xml').is_file()
    assert (wdir / f'{stem}.pdb').is_file()


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ('gaff2', 'gas'),
        ('openff-2.1.0', 'gas'),
    ]
)
def test_parametrize_ligand_with_smiles(forcefield, charge_method, testdir):
    """Test parametrization with SMILES input using LigandLoader and run method."""
    smiles = 'c1ccncc1CCC(=O)N'
    name = 'test_ligand'
    wdir = testdir / f'{forcefield}_{charge_method}_smiles'
    
    # Create ligand from SMILES using LigandLoader
    loader = LigandLoader()
    # Create a temporary SMILES file
    smi_file = wdir / f'{name}.smi'
    smi_file.parent.mkdir(parents=True, exist_ok=True)
    with open(smi_file, 'w') as f:
        f.write(f'{smiles} {name}\n')
    
    ligands = loader.load(smi_file, only_first=True)
    assert len(ligands) == 1
    ligand = ligands[0]
    
    # Run parametrization
    ff = load_parametrizer(forcefield, charge_method)
    ligand = ff.run(ligand)
    
    # Check that auxiliary files are present
    assert 'prmtop' in ligand.auxiliary_files
    assert 'inpcrd' in ligand.auxiliary_files
    
    # Write files to directory for checking
    ligand.dump(wdir)
    
    # Check that required files are generated
    assert (wdir / f'{name}.prmtop').is_file()
    assert (wdir / f'{name}.inpcrd').is_file()
    assert (wdir / f'{name}.sdf').is_file()
    assert 'xml' in ligand.auxiliary_files
    assert 'pdb' in ligand.auxiliary_files


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ('gaff2', 'gas'),
        ('openff-2.1.0', 'gas'),
    ]
)
def test_parametrize_ligand_with_rdkit_mol(forcefield, charge_method, testdir):
    """Test parametrization with RDKit molecule input using LigandLoader and run method."""
    smiles = 'c1ccncc1CCC(=O)N'
    name = 'test_rdkit_mol'
    wdir = testdir / f'{forcefield}_{charge_method}_rdkit_mol'
    
    # Create RDKit molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp('_Name', name)
    
    # Create ligand from RDKit molecule using LigandLoader
    loader = LigandLoader()
    ligands = loader.load([mol])
    assert len(ligands) == 1
    ligand = ligands[0]
    
    # Run parametrization
    ff = load_parametrizer(forcefield, charge_method)
    ligand = ff.run(ligand)
    
    # Check that auxiliary files are present
    assert 'prmtop' in ligand.auxiliary_files
    assert 'inpcrd' in ligand.auxiliary_files
    
    # Write files to directory for checking
    ligand.dump(wdir)
    
    # Check that required files are generated
    assert (wdir / f'{name}.prmtop').is_file()
    assert (wdir / f'{name}.inpcrd').is_file()
    assert (wdir / f'{name}.sdf').is_file()
    assert 'xml' in ligand.auxiliary_files
    assert 'pdb' in ligand.auxiliary_files


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ('gaff2', 'gas'),
        ('openff-2.1.0', 'gas'),
    ]
)
def test_parametrize_ligand_with_rdkit_mol_no_3d(forcefield, charge_method, testdir):
    """Test parametrization with RDKit molecule (no 3D) using LigandLoader and run method."""
    smiles = 'c1ccncc1CCC(=O)N'
    name = 'test_rdkit_mol_no_3d'
    wdir = testdir / f'{forcefield}_{charge_method}_rdkit_mol_no_3d'
    
    # Create RDKit molecule from SMILES without 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    mol.SetProp('_Name', name)
    # Don't embed 3D coordinates - let embed() handle it
    
    # Create ligand from RDKit molecule using LigandLoader
    loader = LigandLoader()
    ligands = loader.load([mol])
    assert len(ligands) == 1
    ligand = ligands[0]
    
    # Run parametrization (embed() will be called automatically)
    ff = load_parametrizer(forcefield, charge_method)
    ligand = ff.run(ligand)
    
    # Check that auxiliary files are present
    assert 'prmtop' in ligand.auxiliary_files
    assert 'inpcrd' in ligand.auxiliary_files
    
    # Write files to directory for checking
    ligand.dump(wdir)
    
    # Check that required files are generated
    assert (wdir / f'{name}.prmtop').is_file()
    assert (wdir / f'{name}.inpcrd').is_file()
    assert (wdir / f'{name}.sdf').is_file()
