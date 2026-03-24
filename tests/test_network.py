from pathlib import Path

import pytest
from rdkit import Chem

from easybfe.network import (
    _check_unique_names,
    load_network_generator,
)


def _data_path(name: str) -> Path:
    return Path(__file__).parent / "data" / name


class _LigandStub:
    def __init__(self, name: str, mol):
        self.name = name
        self._mol = mol

    def get_rdmol(self):
        return self._mol


def _load_tyk2_ligands(n: int = 6):
    suppl = Chem.SDMolSupplier(str(_data_path("tyk2_ligands.sdf")), removeHs=False)
    ligands = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else f"lig_{i}"
        ligands.append(_LigandStub(name, mol))
    if n > 0:
        assert len(ligands) >= n
        return ligands[:n]
    else:
        return ligands


def _assert_valid_edge_list(edges, ligands):
    names = {lig.name for lig in ligands}
    assert isinstance(edges, list)
    assert len(edges) > 0
    covered = set()
    for edge in edges:
        assert isinstance(edge, tuple)
        assert len(edge) == 2
        a, b = edge
        assert a in names
        assert b in names
        covered.add(a)
        covered.add(b)
    assert covered == names


def test_check_unique_names_duplicate():
    ligands = _load_tyk2_ligands(3)
    ligands[1].name = ligands[0].name
    with pytest.raises(ValueError, match="Duplicated ligand names"):
        _check_unique_names(ligands)


def test_load_network_generator_builtin():
    ligands = _load_tyk2_ligands(5)
    center = ligands[0].name
    gen = load_network_generator("star", center=center)
    edges = gen.run(ligands)
    _assert_valid_edge_list(edges, ligands)


def test_load_network_generator_unknown():
    with pytest.raises(NotImplementedError, match="not available"):
        load_network_generator("does_not_exist")


@pytest.mark.parametrize("method", ["minimal_spanning", "lomap", "minimal_redundant"])
def test_openfe_network_generators(method: str):
    pytest.importorskip("openfe")
    ligands = _load_tyk2_ligands(16)
    kwargs = {"progress": False}
    if method == "lomap":
        kwargs = {}
    gen = load_network_generator(method, **kwargs)
    edges = gen.run(ligands)
    _assert_valid_edge_list(edges, ligands)
