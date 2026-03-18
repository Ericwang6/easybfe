import json
from pathlib import Path
import pytest
from rdkit import Chem

from easybfe.core import Ligand
from easybfe.mapping import LazyMCSMapper, LomapAtomMapper, KartografAtomMapper
from easybfe.mapping.base import CustomLigandAtomMapper


def _data_path(name):
    return Path(__file__).parent / "data" / name


# Expected mapping for CDD_1819 -> CDD_1845 (shared by test_mapping and CustomLigandAtomMapper tests)
CDD_1819_1845_MAPPING = {
    0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 9, 8: 10, 9: 11,
    10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19,
    18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 31, 25: 32,
    26: 33, 27: 34, 28: 35, 29: 36, 30: 37, 31: 38, 32: 39, 33: 40,
    34: 41, 35: 42, 36: 43, 37: 44, 38: 45, 39: 46, 40: 47, 41: 48,
    42: 49, 43: 50, 44: 51, 45: 52, 46: 53, 47: 54, 48: 55, 49: 56,
    50: 57, 51: 58, 52: 59, 53: 60, 54: 61,
}


@pytest.mark.parametrize("method", ["lazymcs", "lomap", "kartograf"])
def test_mapping(method):
    ligandA = Ligand.from_file(_data_path("CDD_1819.sdf"))
    ligandB = Ligand.from_file(_data_path("CDD_1845.sdf"))
    if method == "lazymcs":
        mcs_path = _data_path("CDD_1845_south_mcs.sdf")
        mcs_mol = Chem.SDMolSupplier(str(mcs_path))[0]
        mapper = LazyMCSMapper(mcs=mcs_mol)
    elif method == "lomap":
        mapper = LomapAtomMapper()
    else:
        mapper = KartografAtomMapper()

    wdir = Path(__file__).parent / f"_test_mapping/{method}"
    wdir.mkdir(exist_ok=True, parents=True)
    mapping = mapper.run(ligandA, ligandB, wdir)
    assert mapping == CDD_1819_1845_MAPPING
    assert (wdir / "atom_mapping.json").is_file()
    assert (wdir / "atom_mapping.png").is_file()


def test_mapping_custom_ligand_atom_mapper_multiple_mode():
    """CustomLigandAtomMapper with multiple perturbation format (nameA~nameB -> mapping)."""
    ligandA = Ligand.from_file(_data_path("CDD_1819.sdf"))
    ligandB = Ligand.from_file(_data_path("CDD_1845.sdf"))
    key = f"{ligandA.name}~{ligandB.name}"
    data = {key: {str(k): str(v) for k, v in CDD_1819_1845_MAPPING.items()}}
    mapper = CustomLigandAtomMapper(data)
    wdir = Path(__file__).parent / "_test_mapping" / "custom"
    wdir.mkdir(exist_ok=True, parents=True)
    mapping = mapper.run(ligandA, ligandB, wdir)
    assert mapping == CDD_1819_1845_MAPPING
    assert (wdir / "atom_mapping.json").is_file()
    assert (wdir / "atom_mapping.png").is_file()


def test_mapping_custom_ligand_atom_mapper_from_file():
    """CustomLigandAtomMapper loading mapping from a JSON file."""
    ligandA = Ligand.from_file(_data_path("CDD_1819.sdf"))
    ligandB = Ligand.from_file(_data_path("CDD_1845.sdf"))
    key = f"{ligandA.name}~{ligandB.name}"
    data = {key: {str(k): str(v) for k, v in CDD_1819_1845_MAPPING.items()}}
    mapping_file = Path(__file__).parent / "_test_mapping" / "custom_mapping.json"
    mapping_file.parent.mkdir(exist_ok=True, parents=True)
    with open(mapping_file, "w") as f:
        json.dump(data, f, indent=2)
    mapper = CustomLigandAtomMapper(mapping_file)
    wdir = Path(__file__).parent / "_test_mapping" / "custom_file"
    wdir.mkdir(exist_ok=True, parents=True)
    mapping = mapper.run(ligandA, ligandB, wdir)
    assert mapping == CDD_1819_1845_MAPPING
    assert (wdir / "atom_mapping.json").is_file()
    assert (wdir / "atom_mapping.png").is_file()


def test_mapping_kartograf():
    ligandA = Ligand.from_file(_data_path("benzene.sdf"))
    ligandB = Ligand.from_file(_data_path("pyridine.sdf"))
    mapper = KartografAtomMapper(allow_element_change=False)
    wdir = Path(__file__).parent / "_test_mapping"
    wdir.mkdir(exist_ok=True, parents=True)
    mapping = mapper.run(ligandA, ligandB, wdir)
    assert mapping == {
        0: 1, 3: 2, 4: 3, 5: 4, 6: 5,
        7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
    }
    assert (wdir / "atom_mapping.json").is_file()
    assert (wdir / "atom_mapping.png").is_file()


def test_mapping_kartograf_allow_element_change():
    ligandA = Ligand.from_file(_data_path("benzene.sdf"))
    ligandB = Ligand.from_file(_data_path("pyridine.sdf"))
    mapper = KartografAtomMapper(allow_element_change=True)
    wdir = Path(__file__).parent / "_test_mapping_allow_element_change"
    wdir.mkdir(exist_ok=True, parents=True)
    mapping = mapper.run(ligandA, ligandB, wdir)
    assert mapping == {
        0: 1, 2: 0, 3: 2, 4: 3, 5: 4,
        6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
    }
    assert (wdir / "atom_mapping.json").is_file()
    assert (wdir / "atom_mapping.png").is_file()
