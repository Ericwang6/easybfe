"""
Tests for AmberPlainMDConfig and related config field dependencies.

All tests use plain nested dictionaries (strings and basic types only) and
model_validate to exercise the model's ability to coerce inputs to the desired types.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from easybfe.config import AmberPlainMDConfig, PlainMDAnalysisConfig


# -----------------------------------------------------------------------------
# 1. Create model and save without errors
# -----------------------------------------------------------------------------


def test_amber_plain_md_config_create_and_save_minimal():
    """Create AmberPlainMDConfig from minimal dict and save to dict without errors."""
    data = {
        "protein": None,
        "ligand": "lig.pdb",
        "output_dir": "run_lig",
    }
    cfg = AmberPlainMDConfig.model_validate(data)
    assert isinstance(cfg.protein, (type(None), Path))
    assert isinstance(cfg.ligand, Path)
    assert isinstance(cfg.output_dir, Path)
    assert cfg.output_dir == Path("run_lig")
    out = cfg.model_dump()
    assert isinstance(out, dict)
    assert "output_dir" in out
    assert "task_type" in out
    assert "analysis" in out
    out_ser = cfg.model_dump(mode="json")
    assert isinstance(out_ser["output_dir"], str)
    assert out_ser["task_type"] == "ligand"


def test_amber_plain_md_config_create_and_save_full():
    """Create from full dict (nested analysis as dict); save without errors."""
    data = {
        "protein": "pro.pdb",
        "ligand": "lig.pdb",
        "output_dir": "run_complex",
        "task_name": "my_run",
        "analysis": {"rmsd_selection": "backbone", "do_alignment": False},
    }
    cfg = AmberPlainMDConfig.model_validate(data)
    assert isinstance(cfg.protein, Path)
    assert isinstance(cfg.ligand, Path)
    assert isinstance(cfg.analysis, PlainMDAnalysisConfig)
    out = cfg.model_dump()
    assert out["task_name"] == "my_run"
    assert out["task_type"] == "complex"
    assert out["analysis"]["rmsd_selection"] == "backbone"
    assert out["analysis"]["do_alignment"] is False


# -----------------------------------------------------------------------------
# 2. Field dependency: task_name / task_type and *_selection defaults
# -----------------------------------------------------------------------------


def test_task_type_ligand_only():
    """task_type is 'ligand' when protein is None and ligand is set (dict with string path)."""
    data = {"protein": None, "ligand": "l.pdb", "output_dir": "run_lig"}
    cfg = AmberPlainMDConfig.model_validate(data)
    assert cfg.task_type == "ligand"
    assert cfg.analysis.center_selection == "resname MOL"
    assert cfg.analysis.output_selection == "resname MOL"
    assert cfg.analysis.align_selection == "resname MOL"
    assert cfg.analysis.rmsd_selection == "resname MOL"


def test_task_type_protein_only():
    """task_type is 'protein' when ligand is None and protein is set."""
    data = {"protein": "p.pdb", "ligand": None, "output_dir": "run_prot"}
    cfg = AmberPlainMDConfig.model_validate(data)
    assert cfg.task_type == "protein"
    assert cfg.analysis.center_selection == "protein"
    assert cfg.analysis.output_selection == "protein or resname MOL"
    assert cfg.analysis.align_selection == "backbone"
    assert cfg.analysis.rmsd_selection == "backbone"


def test_task_type_complex():
    """task_type is 'complex' when both protein and ligand are set (string paths)."""
    data = {"protein": "p.pdb", "ligand": "l.pdb", "output_dir": "run_complex"}
    cfg = AmberPlainMDConfig.model_validate(data)
    assert cfg.task_type == "complex"
    assert cfg.analysis.center_selection == "protein"
    assert cfg.analysis.output_selection == "protein or resname MOL"
    assert cfg.analysis.align_selection == "backbone"
    assert cfg.analysis.rmsd_selection == "resname MOL"


def test_task_name_default_from_output_dir():
    """task_name defaults to output_dir.name when not specified (output_dir as string)."""
    data = {"protein": "p.pdb", "ligand": None, "output_dir": "/some/path/my_simulation"}
    cfg = AmberPlainMDConfig.model_validate(data)
    assert isinstance(cfg.output_dir, Path)
    assert cfg.task_name == "my_simulation"
    assert cfg.analysis.rmsd_name == "my_simulation"


def test_task_name_user_specified():
    """task_name stays user-specified when provided in dict."""
    data = {
        "protein": "p.pdb",
        "ligand": "l.pdb",
        "output_dir": "run_complex",
        "task_name": "custom_task",
    }
    cfg = AmberPlainMDConfig.model_validate(data)
    assert cfg.task_name == "custom_task"
    assert cfg.analysis.rmsd_name == "custom_task"


def test_analysis_user_selection_overrides_inferred():
    """User-specified *_selection in nested analysis dict overrides inferred defaults."""
    data = {
        "protein": "p.pdb",
        "ligand": None,
        "output_dir": "run_prot",
        "analysis": {
            "rmsd_selection": "name CA",
            "center_selection": "protein and name CA",
            "align_selection": "name CA",
        },
    }
    cfg = AmberPlainMDConfig.model_validate(data)
    assert cfg.task_type == "protein"
    assert cfg.analysis.rmsd_selection == "name CA"
    assert cfg.analysis.center_selection == "protein and name CA"
    assert cfg.analysis.align_selection == "name CA"
    assert cfg.analysis.output_selection == "protein or resname MOL"


def test_analysis_partial_user_spec_rest_inferred():
    """Only specified analysis fields override; rest come from task_type/task_name."""
    data = {
        "protein": None,
        "ligand": "l.pdb",
        "output_dir": "run_lig",
        "task_name": "mylig",
        "analysis": {"rmsd_selection": "resname MOL and name C*"},
    }
    cfg = AmberPlainMDConfig.model_validate(data)
    assert cfg.analysis.rmsd_selection == "resname MOL and name C*"
    assert cfg.analysis.center_selection == "resname MOL"
    assert cfg.analysis.rmsd_name == "mylig"


def test_amber_plain_md_config_both_none_raises():
    """model_validate with both protein and ligand None raises."""
    data = {"protein": None, "ligand": None, "output_dir": "run"}
    with pytest.raises(ValidationError, match="Protein and Ligand cannot be None"):
        AmberPlainMDConfig.model_validate(data)


# -----------------------------------------------------------------------------
# 3. Idempotency: dump -> model_validate round-trip
# -----------------------------------------------------------------------------


def test_amber_plain_md_config_idempotency_minimal():
    """Round-trip from minimal dict: model_validate -> dump -> model_validate yields equal model."""
    data = {"protein": None, "ligand": "lig.pdb", "output_dir": "run_lig"}
    original = AmberPlainMDConfig.model_validate(data)
    dumped = original.model_dump()
    restored = AmberPlainMDConfig.model_validate(dumped)
    assert restored == original


def test_amber_plain_md_config_idempotency_full():
    """Round-trip from full dict (nested analysis) yields equal model."""
    data = {
        "protein": "pro.pdb",
        "ligand": "lig.pdb",
        "output_dir": "out/complex_run",
        "task_name": "my_complex",
        "analysis": {
            "rmsd_selection": "backbone",
            "center_selection": "protein",
            "rmsd_name": "custom_rmsd",
            "do_gbsa": False,
        },
    }
    original = AmberPlainMDConfig.model_validate(data)
    dumped = original.model_dump()
    restored = AmberPlainMDConfig.model_validate(dumped)
    assert restored == original


def test_amber_plain_md_config_idempotency_json_mode():
    """Round-trip via model_dump(mode='json') then model_validate yields equal model (paths as strings)."""
    data = {
        "protein": "a/pro.pdb",
        "ligand": "b/lig.pdb",
        "output_dir": "out/run",
        "task_name": "json_roundtrip",
    }
    original = AmberPlainMDConfig.model_validate(data)
    dumped = original.model_dump(mode="json")
    assert isinstance(dumped["output_dir"], str)
    restored = AmberPlainMDConfig.model_validate(dumped)
    assert restored.task_type == original.task_type
    assert restored.task_name == original.task_name
    assert restored.protein == original.protein
    assert restored.ligand == original.ligand
    assert restored.output_dir == original.output_dir
    assert restored.analysis.rmsd_name == original.analysis.rmsd_name
    assert isinstance(restored.protein, Path)
    assert isinstance(restored.output_dir, Path)
