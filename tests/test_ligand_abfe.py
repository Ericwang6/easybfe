import os
import shutil
from pathlib import Path

from easybfe.core.ligand import LigandLoader
from easybfe.core.protein import Protein
from easybfe.config import AmberFepSimulationConfig
from easybfe.config.amber.simulation import default_abfe_workflow
from easybfe.amber.prep_ligand_abfe import setup_ligand_abfe, BoreschRestraint
from easybfe.analysis.abfe import analyze_abfe
from easybfe.smff import load_parametrizer


# Boresch
# protein: [1427, 1412, 1155]
# ligand: [14, 12, 11]
# rst_wt: [10.0, 100.0, 100.0, 100.0, 100.0, 100.0]
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
    ligands = loader.load(ligand_sdf, only_first=True, use_stem_as_name=True)
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
    
    # Create Boresch restraint
    # Note: indices are 0-based in Python, so subtract 1 from comment values (which are 1-based PDB indices)
    protein_anchors = (1448, 1450, 1451)  # Convert to 0-based: [1426, 1411, 1154]
    ligand_anchors = (14, 15, 16)  # Convert to 0-based: [13, 11, 10]
    rst_wts = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    restraints = BoreschRestraint(
        protein_anchors=protein_anchors,
        ligand_anchors=ligand_anchors,
        rst_wts=rst_wts
    )
    
    # Create configs for each leg
    # Use minimal lambdas for faster testing and the default ABFE workflow
    leg_configs = {
        'solvent': AmberFepSimulationConfig(num_lambdas=16, workflow=default_abfe_workflow()),
        'complex': AmberFepSimulationConfig(num_lambdas=16, workflow=default_abfe_workflow()),
        'restraint': AmberFepSimulationConfig(num_lambdas=16, workflow=default_abfe_workflow()),
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


def test_ligand_abfe_analysis():
    test_dir = Path(__file__).parent / '_test_ligand_abfe_old/abfe_output'
    analyze_abfe(test_dir, '05.prod')
