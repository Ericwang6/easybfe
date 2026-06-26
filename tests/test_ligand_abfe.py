import os
import shutil
import logging
from pathlib import Path

from easybfe.core.ligand import LigandLoader
from easybfe.core.protein import Protein
from easybfe.config import AmberFepSimulationConfig
from easybfe.config.amber.simulation import default_fep_workflow
from easybfe.amber.prep_ligand_abfe import setup_ligand_abfe
from easybfe.boresch import RxRxBoreschRestraintsFinder
from easybfe.analysis.abfe import analyze_abfe
from easybfe.smff import load_parametrizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def test_setup_ligand_abfe():
    """Test setup_ligand_abfe function with Boresch restraints."""
    # Setup test directory
    test_dir = Path(__file__).parent / '_test_ligand_abfe'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Get test data paths
    data_dir = Path(__file__).parent / 'data'
    ligand_sdf = data_dir / 'jmc_23.sdf'
    protein_pdb = data_dir / 'tyk2_pdbfixer.pdb'
    
    # Load and parametrize ligand
    loader = LigandLoader()
    ligands = loader.load(ligand_sdf, only_first=True, name_from_stem=True)
    assert len(ligands) == 1
    ligand = ligands[0]
    
    # Parametrize ligand (this adds xml, pdb, prmtop, inpcrd to auxiliary_files)
    smff = load_parametrizer('gaff2', 'gas')
    ligand = smff.run(ligand)
    
    # Verify ligand has required auxiliary files
    assert 'xml' in ligand.auxiliary_files
    assert 'pdb' in ligand.auxiliary_files
    
    # Load protein
    protein = Protein.from_pdb(protein_pdb, name='tyk2')
    
    # Automatically find Boresch restraints from protein–ligand geometry
    finder = RxRxBoreschRestraintsFinder(protein, ligand)
    restraints = finder.find()
    
    # Create configs for each leg
    # Use minimal lambdas for faster testing and the default ABFE workflow
    leg_configs = {
        'solvent': AmberFepSimulationConfig(num_lambdas=16, workflow=default_fep_workflow()),
        'complex': AmberFepSimulationConfig(num_lambdas=16, workflow=default_fep_workflow()),
        'restraint': AmberFepSimulationConfig(num_lambdas=16, workflow=default_fep_workflow()),
    }
    
    # Setup ABFE
    output_dir = test_dir / 'abfe_output'
    setup_ligand_abfe(
        ligand=ligand,
        protein=protein,
        leg_configs=leg_configs,
        restraints=restraints,
        output_dir=output_dir
    )
    
    # Verify output directories were created
    assert (output_dir / 'solvent').exists()
    assert (output_dir / 'complex').exists()
    assert (output_dir / 'restraint').exists()
    
    # Verify key files exist in each leg
    for leg_name in ['solvent', 'complex', 'restraint']:
        leg_dir = output_dir / leg_name
        assert (leg_dir / 'system.prmtop').exists()
        assert (leg_dir / 'system.inpcrd').exists()
        assert (leg_dir / 'system.pdb').exists()
        assert (leg_dir / f'{ligand.name}.pdb').exists()
        assert (leg_dir / f'{ligand.name}.xml').exists()
        # Verify lambda directories exist
        assert (leg_dir / 'lambda0').exists()
        assert (leg_dir / 'lambda1').exists()
        assert (leg_dir / 'lambda2').exists()
        # Verify run.sh exists in each leg directory
        assert (leg_dir / 'run.sh').exists()
    
    # Cleanup
    # shutil.rmtree(test_dir)


def test_setup_ligand_abfe_charge_change():
    """Test setup_ligand_abfe with different charge-change methods."""
    # Setup test directory
    test_dir = Path(__file__).parent / "_test_ligand_abfe_charge_change"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Get test data paths
    data_dir = Path(__file__).parent / "data"
    ligand_sdf = data_dir / "jmc_32.sdf"
    protein_pdb = data_dir / "tyk2_pdbfixer.pdb"

    # Load and parametrize ligand
    loader = LigandLoader()
    ligands = loader.load(ligand_sdf, only_first=True, name_from_stem=True)
    assert len(ligands) == 1
    ligand = ligands[0]

    smff = load_parametrizer("gaff2", "gas")
    ligand = smff.run(ligand)

    # Load protein
    protein = Protein.from_pdb(protein_pdb, name="tyk2")

    # Charge-change methods to test
    methods = ["dummy_ion", "coalchem_water"]

    for method in methods:
        # Create configs for each leg with specified charge-change method
        leg_configs = {
            "solvent": AmberFepSimulationConfig(
                num_lambdas=8,
                workflow=default_fep_workflow(),
                charge_change_method=method,
            ),
            "complex": AmberFepSimulationConfig(
                num_lambdas=8,
                workflow=default_fep_workflow(),
                charge_change_method=method,
            ),
            "restraint": AmberFepSimulationConfig(
                num_lambdas=8,
                workflow=default_fep_workflow(),
                charge_change_method=method,
            ),
        }

        output_dir = test_dir / f"abfe_output_{method}"

        setup_ligand_abfe(
            ligand=ligand,
            protein=protein,
            leg_configs=leg_configs,
            restraints=None,
            output_dir=output_dir,
        )

        # Verify output directories and key files were created
        for leg_name in ["solvent", "complex", "restraint"]:
            leg_dir = output_dir / leg_name
            assert leg_dir.exists()
            assert (leg_dir / "system.prmtop").exists()
            assert (leg_dir / "system.inpcrd").exists()
            assert (leg_dir / "system.pdb").exists()
            assert (leg_dir / f"{ligand.name}.pdb").exists()
            assert (leg_dir / f"{ligand.name}.xml").exists()
            # Verify at least a few lambda directories exist
            assert (leg_dir / "lambda0").exists()
            assert (leg_dir / "lambda1").exists()
            assert (leg_dir / "lambda2").exists()
            # Verify run.sh exists in each leg directory
            assert (leg_dir / "run.sh").exists()

