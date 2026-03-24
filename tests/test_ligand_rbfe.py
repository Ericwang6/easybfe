import json
import logging
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
import easybfe.amber.prep_ligand_rbfe as prep_ligand_rbfe
from easybfe.core.ligand import LigandLoader
from easybfe.core.protein import Protein
from easybfe.config import AmberFepSimulationConfig
from easybfe.config.amber.rbfe import AmberLigandRbfeConfig
from easybfe.config.amber.simulation import default_fep_workflow
from easybfe.amber.prep_ligand_rbfe import setup_ligand_rbfe
from easybfe.analysis.rbfe import analyze_rbfe, analyze_rbfe_dg_network
from easybfe.smff import load_parametrizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _load_and_parametrize(sdf_path: Path):
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
        "complex": AmberFepSimulationConfig(num_lambdas=8, workflow=default_fep_workflow()),
        "solvent": AmberFepSimulationConfig(num_lambdas=8, workflow=default_fep_workflow()),
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
            workflow=default_fep_workflow(),
            charge_change_method="dummy_ion",
        ),
        "solvent": AmberFepSimulationConfig(
            num_lambdas=8,
            workflow=default_fep_workflow(),
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


def test_analyze_rbfe_with_extracted_archive(tmp_path: Path):
    data_dir = Path(__file__).parent / "data"
    archive_path = data_dir / "ejm_44~ejm_31.tar.gz"
    extract_root = data_dir / "_test_rbfe_data"
    if extract_root.is_dir():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(extract_root, filter="data")

    wdir = extract_root / "ejm_44~ejm_31"
    assert wdir.exists()

    for leg in ["complex", "solvent"]:
        (wdir / leg / "done.tag").touch()

    result = analyze_rbfe(wdir, prod_prefix="05.prod", force_run=True)

    assert "dg_complex" in result
    assert "dg_complex_std" in result
    assert "dg_solvent" in result
    assert "dg_solvent_std" in result
    assert "ddg_total" in result
    assert "ddg_total_std" in result
    assert isinstance(result["ddg_total"], float)
    assert isinstance(result["ddg_total_std"], float)
    assert result["ddg_total_std"] >= 0.0

    assert (wdir / "result.json").exists()
    assert (wdir / "total_convergence.csv").exists()
    assert (wdir / "total_convergence.png").exists()


def test_analyze_rbfe_dg_network_writes_dg_csv(tmp_path: Path):
    batch = tmp_path / "batch"
    batch.mkdir()
    for sub, ddg, std in [
        ("A~B", 1.0, 0.1),
        ("B~C", 0.5, 0.1),
        ("A~C", 1.5, 0.1),
    ]:
        d = batch / sub
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps({"ddg_total": ddg, "ddg_total_std": std})
        )

    out = analyze_rbfe_dg_network(batch, bias=0.0)
    assert (batch / "dg.csv").is_file()
    assert set(out.columns) == {"ligand", "dG", "dG_std"}
    assert len(out) == 3

    # Missing result.json is skipped; MLE still runs on remaining edges.
    (batch / "X~Y").mkdir()
    out2 = analyze_rbfe_dg_network(batch, bias=0.0)
    assert len(out2) == 3


def test_setup_ligand_rbfe_from_config_network_mode(tmp_path: Path, monkeypatch):
    lig_base = tmp_path / "ligands"
    lig_base.mkdir()
    protein = tmp_path / "protein.pdb"
    protein.write_text("HEADER\n")
    output_base = tmp_path / "rbfe"
    captured = {}

    class DummyGenerator:
        def run(self, ligands):
            captured["ligand_names"] = [lig.name for lig in ligands]
            return [("ejm_44", "ejm_31")]

    monkeypatch.setattr(
        prep_ligand_rbfe.Protein, "from_pdb", lambda *args, **kwargs: SimpleNamespace(name="protein")
    )
    monkeypatch.setattr(
        prep_ligand_rbfe.Ligand, "from_directory", lambda p: SimpleNamespace(name=Path(p).name)
    )
    monkeypatch.setattr(prep_ligand_rbfe, "load_network_generator", lambda *args, **kwargs: DummyGenerator())
    monkeypatch.setattr(
        prep_ligand_rbfe,
        "run_func_parallel",
        lambda func, args_list, nprocs, desc: captured.update({"args_list": args_list, "nprocs": nprocs, "desc": desc}),
    )

    cfg = AmberLigandRbfeConfig.model_validate(
        {
            "protein": protein,
            "ligand_base": lig_base,
            "ligand_list": ["ejm_44", "ejm_31"],
            "network": {"algorithm": "custom", "options": {"edges": [["ejm_44", "ejm_31"]]}},
            "output_base": output_base,
            "ligandA": "should_be_ignored",
            "ligandB": "should_be_ignored",
        }
    )
    prep_ligand_rbfe.setup_ligand_rbfe_from_config(cfg, num_procs=2)
    args = captured["args_list"]
    assert len(args) == 1
    lig_a_dir, lig_b_dir, _, _, _, out_dir = args[0]
    assert lig_a_dir == (lig_base / "ejm_44").resolve()
    assert lig_b_dir == (lig_base / "ejm_31").resolve()
    assert out_dir == (output_base / "ejm_44~ejm_31").resolve()
    assert captured["ligand_names"] == ["ejm_44", "ejm_31"]
    assert captured["nprocs"] == 2
    assert captured["desc"] == "setup_ligand_rbfe"


def test_setup_ligand_rbfe_from_config_pair_mode(tmp_path: Path, monkeypatch):
    lig_base = tmp_path / "ligands"
    lig_base.mkdir()
    protein = tmp_path / "protein.pdb"
    protein.write_text("HEADER\n")
    output_base = tmp_path / "rbfe"
    captured = {}

    monkeypatch.setattr(
        prep_ligand_rbfe.Protein, "from_pdb", lambda *args, **kwargs: SimpleNamespace(name="protein")
    )
    monkeypatch.setattr(
        prep_ligand_rbfe, "_setup_ligand_rbfe_one", lambda job: captured.update({"job": job})
    )

    cfg = AmberLigandRbfeConfig.model_validate(
        {
            "protein": protein,
            "ligand_base": lig_base,
            "ligandA": "ejm_44",
            "ligandB": "ejm_31",
            "output_base": output_base,
            "ligand_list": None,
        }
    )
    prep_ligand_rbfe.setup_ligand_rbfe_from_config(cfg)
    lig_a_dir, lig_b_dir, _, _, _, out_dir = captured["job"]
    assert lig_a_dir == (lig_base / "ejm_44").resolve()
    assert lig_b_dir == (lig_base / "ejm_31").resolve()
    assert out_dir == (output_base / "ejm_44~ejm_31").resolve()


def test_rbfe_config_requires_valid_input_mode(tmp_path: Path):
    protein = tmp_path / "protein.pdb"
    protein.write_text("HEADER\n")
    with pytest.raises(ValidationError):
        AmberLigandRbfeConfig.model_validate({"protein": protein})
