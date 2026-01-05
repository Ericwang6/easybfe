import shutil
from pathlib import Path
import pytest
from easybfe.smff import parametrize_ligand, load_parametrizer


@pytest.fixture
def testdir():
    """Create and clean up test directory."""
    testdir = Path(__file__).parent / '_test_smff'
    if testdir.is_dir():
        shutil.rmtree(testdir)
    testdir.mkdir()
    yield testdir
    if testdir.is_dir():
        shutil.rmtree(testdir)


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
    """Test parametrize_ligand with file input."""
    wdir = testdir / f'{forcefield}_{charge_method}_file'
    parametrize_ligand(amide_sdf, wdir, forcefield, charge_method)
    
    # Check that required files are generated
    stem = amide_sdf.stem
    assert (wdir / f'{stem}.prmtop').is_file()
    assert (wdir / f'{stem}.inpcrd').is_file()
    assert (wdir / f'{stem}.sdf').is_file()
    assert (wdir / f'{stem}.xml').is_file()
    assert (wdir / f'{stem}.pdb').is_file()
    assert (wdir / f'{stem}.png').is_file()


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ('gaff2', 'gas'),
        ('openff-2.1.0', 'gas'),
    ]
)
def test_parametrize_ligand_with_smiles(forcefield, charge_method, testdir):
    """Test parametrize_ligand with SMILES input."""
    smiles = 'c1ccncc1CCC(=O)N'
    name = 'test_ligand'
    wdir = testdir / f'{forcefield}_{charge_method}_smiles'
    
    # Use parameterizer directly to pass name parameter
    ff = load_parametrizer(forcefield, charge_method)
    ff.run(smiles, wdir, name=name, overwrite=False)
    
    # Check that required files are generated
    assert (wdir / f'{name}.prmtop').is_file()
    assert (wdir / f'{name}.inpcrd').is_file()
    assert (wdir / f'{name}.sdf').is_file()
    assert (wdir / f'{name}.xml').is_file()
    assert (wdir / f'{name}.pdb').is_file()
    assert (wdir / f'{name}.png').is_file()


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ('gaff2', 'gas'),
        ('openff-2.1.0', 'gas'),
    ]
)
def test_parametrize_ligand_overwrite_false_raises_error(forcefield, charge_method, testdir, amide_sdf):
    """Test that an error is raised when directory exists and overwrite=False."""
    wdir = testdir / f'{forcefield}_{charge_method}_overwrite'
    
    # First run - should succeed
    ff = load_parametrizer(forcefield, charge_method)
    ff.run(amide_sdf, wdir, overwrite=False)
    
    # Second run with overwrite=False - should raise FileExistsError
    with pytest.raises(FileExistsError):
        ff.run(amide_sdf, wdir, overwrite=False)


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ('gaff2', 'gas'),
        ('openff-2.1.0', 'gas'),
    ]
)
def test_parametrize_ligand_overwrite_true(forcefield, charge_method, testdir, amide_sdf):
    """Test that overwrite=True allows re-running parametrization."""
    wdir = testdir / f'{forcefield}_{charge_method}_overwrite_true'
    
    # First run
    ff = load_parametrizer(forcefield, charge_method)
    ff.run(amide_sdf, wdir, overwrite=False)
    
    # Second run with overwrite=True - should succeed
    ff.run(amide_sdf, wdir, overwrite=True)
    
    # Check that files still exist
    stem = amide_sdf.stem
    assert (wdir / f'{stem}.prmtop').is_file()
    assert (wdir / f'{stem}.inpcrd').is_file()
