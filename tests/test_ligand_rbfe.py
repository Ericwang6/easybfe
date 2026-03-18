import shutil
import logging
from pathlib import Path

from easybfe.core.ligand import LigandLoader
from easybfe.core.protein import Protein
from easybfe.config import AmberFepSimulationConfig, default_md_workflow
from easybfe.amber.prep_ligand_rbfe import setup_ligand_rbfe
from easybfe.smff import load_parametrizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _load_and_parametrize(sdf_path: Path) -> "Ligand":
    loader = LigandLoader()
    ligands = loader.load(sdf_path, only_first=True, name_from_stem=True)
    assert len(ligands) == 1
    ligand = ligands[0]
    smff = load_parametrizer("gaff2", "gas")
    ligand = smff.run(ligand)
    assert "xml" in ligand.auxiliary_files
    assert "pdb" in ligand.auxiliary_files
    return ligand


def test_setup_ligand_rbfe_normal():
    """Test RBFE setup for a normal (no charge change) perturbation: jmc_23 -> ejm_47."""
    test_dir = Path(__file__).parent / "_test_ligand_rbfe_normal"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    data_dir = Path(__file__).parent / "data"

    ligandA = _load_and_parametrize(data_dir / "jmc_23.sdf")
    ligandB = _load_and_parametrize(data_dir / "ejm_47.sdf")
    protein = Protein.from_pdb(data_dir / "tyk2_pdbfixer.pdb", name="tyk2")

    leg_configs = {
        "complex": AmberFepSimulationConfig(num_lambdas=8, workflow=default_md_workflow()),
        "solvent": AmberFepSimulationConfig(num_lambdas=8, workflow=default_md_workflow()),
    }

    output_dir = test_dir / "rbfe_output"
    setup_ligand_rbfe(
        ligandA=ligandA,
        ligandB=ligandB,
        mapping=None,
        protein=protein,
        leg_configs=leg_configs,
        output_dir=output_dir,
    )

    assert (output_dir / "atom_mapping.json").exists()
    for leg_name in ["complex", "solvent"]:
        leg_dir = output_dir / leg_name
        assert leg_dir.exists(), f"{leg_name} directory not found"
        assert (leg_dir / "system.prmtop").exists()
        assert (leg_dir / "system.inpcrd").exists()
        assert (leg_dir / "system.pdb").exists()
        assert (leg_dir / "lambda0").exists()
        assert (leg_dir / "lambda1").exists()
        assert (leg_dir / "run.sh").exists()


def test_setup_ligand_rbfe_charge_change():
    """Test RBFE setup for a charge-changing perturbation: jmc_23 -> jmc_32."""
    test_dir = Path(__file__).parent / "_test_ligand_rbfe_cc"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    data_dir = Path(__file__).parent / "data"

    ligandA = _load_and_parametrize(data_dir / "jmc_23.sdf")
    ligandB = _load_and_parametrize(data_dir / "jmc_32.sdf")
    protein = Protein.from_pdb(data_dir / "tyk2_pdbfixer.pdb", name="tyk2")

    leg_configs = {
        "complex": AmberFepSimulationConfig(
            num_lambdas=8,
            workflow=default_md_workflow(),
            charge_change_method="dummy_ion",
        ),
        "solvent": AmberFepSimulationConfig(
            num_lambdas=8,
            workflow=default_md_workflow(),
            charge_change_method="dummy_ion",
        ),
    }

    output_dir = test_dir / "rbfe_output"
    setup_ligand_rbfe(
        ligandA=ligandA,
        ligandB=ligandB,
        mapping=None,
        protein=protein,
        leg_configs=leg_configs,
        output_dir=output_dir,
    )

    assert (output_dir / "atom_mapping.json").exists()
    for leg_name in ["complex", "solvent"]:
        leg_dir = output_dir / leg_name
        assert leg_dir.exists(), f"{leg_name} directory not found"
        assert (leg_dir / "system.prmtop").exists()
        assert (leg_dir / "system.inpcrd").exists()
        assert (leg_dir / "system.pdb").exists()
        assert (leg_dir / "lambda0").exists()
        assert (leg_dir / "lambda1").exists()
        assert (leg_dir / "run.sh").exists()
