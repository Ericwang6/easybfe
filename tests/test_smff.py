import shutil
from pathlib import Path
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from easybfe.smff import (
    PARAMETRIZER_REGISTRY,
    load_parametrizer,
    parametrize_ligands,
)
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
    if forcefield.startswith('openff') and 'openff' not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")
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
    
    # Check that required files are generated, using the ligand name actually used
    name = ligand.name
    assert (wdir / f'{name}.prmtop').is_file()
    assert (wdir / f'{name}.inpcrd').is_file()
    assert (wdir / f'{name}.sdf').is_file()
    assert (wdir / f'{name}.xml').is_file()
    assert (wdir / f'{name}.pdb').is_file()


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ('gaff2', 'gas'),
        ('openff-2.1.0', 'gas'),
    ]
)
def test_parametrize_ligand_with_smiles(forcefield, charge_method, testdir):
    """Test parametrization with SMILES input using LigandLoader and run method."""
    if forcefield.startswith('openff') and 'openff' not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")
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
    if forcefield.startswith('openff') and 'openff' not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")
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
    if forcefield.startswith('openff') and 'openff' not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")
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


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ("gaff2", "gas"),
    ],
)
def test_parametrize_ligands_single_file(forcefield, charge_method, testdir, amide_sdf):
    """Test parametrize_ligands with a single-ligand SDF source."""
    if forcefield.startswith("openff") and "openff" not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")

    out_base = testdir / f"{forcefield}_{charge_method}_param_ligands_single"
    ligands = parametrize_ligands(
        source=amide_sdf,
        output_base_dir=out_base,
        forcefield=forcefield,
        charge_method=charge_method,
        nprocs=1,
        only_first=True,
        name_from_stem=True,
    )

    assert isinstance(ligands, list)
    assert len(ligands) == 1
    lig = ligands[0]
    assert "prmtop" in lig.auxiliary_files
    assert "inpcrd" in lig.auxiliary_files
    assert "xml" in lig.auxiliary_files
    assert "pdb" in lig.auxiliary_files

    # Single loaded ligand: outputs directly under base (no per-name subdir)
    wdir = out_base
    assert (wdir / f"{lig.name}.prmtop").is_file()
    assert (wdir / f"{lig.name}.inpcrd").is_file()
    assert (wdir / f"{lig.name}.sdf").is_file()
    assert (wdir / f"{lig.name}.xml").is_file()
    assert (wdir / f"{lig.name}.pdb").is_file()


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ("gaff2", "gas"),
    ],
)
def test_parametrize_ligands_multiple_rdkit(forcefield, charge_method, testdir):
    """Test parametrize_ligands with multiple RDKit molecules and name handling."""
    if forcefield.startswith("openff") and "openff" not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")

    smiles_list = ["c1ccncc1CCC(=O)N", "CCO"]
    names = ["ligand_a", "ligand_b"]
    mols = []
    for smi, name in zip(smiles_list, names):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("_Name", name)
        mols.append(mol)

    out_base = testdir / f"{forcefield}_{charge_method}_param_ligands_multi"
    ligands = parametrize_ligands(
        source=mols,
        output_base_dir=out_base,
        forcefield=forcefield,
        charge_method=charge_method,
        nprocs=2,
    )

    assert isinstance(ligands, list)
    assert len(ligands) == len(mols)

    # Names should be preserved and each ligand should have files
    returned_names = {lig.name for lig in ligands}
    assert set(names).issubset(returned_names)

    for lig in ligands:
        assert "prmtop" in lig.auxiliary_files
        assert "inpcrd" in lig.auxiliary_files
        assert "xml" in lig.auxiliary_files
        assert "pdb" in lig.auxiliary_files

        wdir = out_base / lig.name
        assert (wdir / f"{lig.name}.prmtop").is_file()
        assert (wdir / f"{lig.name}.inpcrd").is_file()
        assert (wdir / f"{lig.name}.sdf").is_file()
        assert (wdir / f"{lig.name}.xml").is_file()
        assert (wdir / f"{lig.name}.pdb").is_file()


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ("gaff2", "gas"),
    ],
)
def test_parametrize_ligands_duplicate_names(forcefield, charge_method, testdir):
    """Test that parametrize_ligands surfaces duplicate-name errors from LigandLoader."""
    if forcefield.startswith("openff") and "openff" not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")

    smiles = "c1ccncc1CCC(=O)N"
    name = "dup_name"
    mols = []
    for _ in range(2):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("_Name", name)
        mols.append(mol)

    out_base = testdir / f"{forcefield}_{charge_method}_param_ligands_dup"
    with pytest.raises(ValueError):
        parametrize_ligands(
            source=mols,
            output_base_dir=out_base,
            forcefield=forcefield,
            charge_method=charge_method,
            nprocs=2,
        )


@pytest.mark.parametrize(
    "forcefield, charge_method",
    [
        ("gaff2", "gas"),
    ],
)
def test_parametrize_ligands_multiple_sdf(forcefield, charge_method, testdir, amide_sdf):
    """Test parametrize_ligands with multiple SDF files and name_from_stem=True."""
    if forcefield.startswith("openff") and "openff" not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")

    data_dir = Path(__file__).parent / "data"
    benzene_sdf = data_dir / "benzene.sdf"

    out_base = testdir / f"{forcefield}_{charge_method}_param_ligands_multi_sdf"
    ligands = parametrize_ligands(
        source=[amide_sdf, benzene_sdf],
        output_base_dir=out_base,
        forcefield=forcefield,
        charge_method=charge_method,
        nprocs=1,
        name_from_stem=True,
        only_first=True,
    )

    # We expect one ligand per file, named from the file stem
    assert isinstance(ligands, list)
    assert len(ligands) == 2
    names = {lig.name for lig in ligands}
    assert {"amide", "benzene"} == names

    for lig in ligands:
        # Parameters should be generated
        assert "prmtop" in lig.auxiliary_files
        assert "inpcrd" in lig.auxiliary_files
        assert "xml" in lig.auxiliary_files
        assert "pdb" in lig.auxiliary_files

        # Files should be written under base_dir / ligand.name
        wdir = out_base / lig.name
        assert (wdir / f"{lig.name}.prmtop").is_file()
        assert (wdir / f"{lig.name}.inpcrd").is_file()
        assert (wdir / f"{lig.name}.sdf").is_file()
        assert (wdir / f"{lig.name}.xml").is_file()
        assert (wdir / f"{lig.name}.pdb").is_file()
