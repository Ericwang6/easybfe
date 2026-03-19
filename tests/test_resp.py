import shutil
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from easybfe.smff.resp import (
    parse_resp_spec,
    generate_qchem_input,
    parse_resp_charges,
)
from easybfe.smff import load_parametrizer, PARAMETRIZER_REGISTRY
from easybfe.core.ligand import LigandLoader

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Unit tests for resp.py helpers
# ---------------------------------------------------------------------------

class TestParseRespSpec:
    def test_default(self):
        func, basis = parse_resp_spec("resp")
        assert func == "HF"
        assert basis == "6-31G*"

    def test_custom(self):
        func, basis = parse_resp_spec("resp/B3LYP/6-311G**")
        assert func == "B3LYP"
        assert basis == "6-311G**"

    def test_invalid_prefix(self):
        with pytest.raises(ValueError):
            parse_resp_spec("bcc")

    def test_invalid_parts(self):
        with pytest.raises(ValueError):
            parse_resp_spec("resp/HF")


class TestGenerateQchemInput:
    @pytest.fixture
    def benzene_mol(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def test_structure(self, benzene_mol):
        inp = generate_qchem_input(benzene_mol, "HF", "6-31G*")
        assert "$molecule" in inp
        assert "$end" in inp
        assert "$rem" in inp
        assert "RESP_CHARGES" in inp
        assert "METHOD               HF" in inp
        assert "BASIS                6-31G*" in inp
        assert "0 1" in inp

    def test_atom_count(self, benzene_mol):
        inp = generate_qchem_input(benzene_mol, "HF", "6-31G*")
        mol_section = inp.split("$molecule")[1].split("$end")[0]
        coord_lines = [
            l for l in mol_section.strip().splitlines()
            if l.strip() and l.strip() != "0 1"
        ]
        assert len(coord_lines) == 12  # 6C + 6H

    def test_custom_functional_basis(self, benzene_mol):
        inp = generate_qchem_input(benzene_mol, "B3LYP", "cc-pVDZ")
        assert "METHOD               B3LYP" in inp
        assert "BASIS                cc-pVDZ" in inp


class TestParseRespCharges:
    def test_benzene_fixture(self):
        out_path = DATA_DIR / "benzene_resp_qchem.out"
        charges = parse_resp_charges(out_path)
        assert len(charges) == 12
        assert abs(sum(charges)) < 1e-4
        for q in charges[:6]:
            assert abs(q - (-0.138152)) < 0.01
        for q in charges[6:]:
            assert abs(q - 0.138152) < 0.01

    def test_missing_section(self, tmp_path):
        fake = tmp_path / "empty.out"
        fake.write_text("no RESP section here\n")
        with pytest.raises(RuntimeError, match="Could not find"):
            parse_resp_charges(fake)


# ---------------------------------------------------------------------------
# End-to-end integration tests (require qchem + acpype / openff)
# ---------------------------------------------------------------------------

@pytest.fixture
def testdir():
    d = Path(__file__).parent / "_test_resp"
    d.mkdir(exist_ok=True)
    yield d


@pytest.fixture
def benzene_sdf():
    return DATA_DIR / "benzene.sdf"


def _assert_resp_charges(ligand, atol=0.02):
    """Check that the charges in the prmtop look like benzene RESP charges."""
    import parmed, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        prmtop = os.path.join(td, "mol.prmtop")
        inpcrd = os.path.join(td, "mol.inpcrd")
        Path(prmtop).write_text(ligand.auxiliary_files["prmtop"])
        Path(inpcrd).write_text(ligand.auxiliary_files["inpcrd"])
        struct = parmed.load_file(prmtop, xyz=inpcrd)

    charges = [a.charge for a in struct.atoms]
    assert abs(sum(charges)) < 0.01
    c_charges = [q for a, q in zip(struct.atoms, charges) if a.element_name == "C"]
    h_charges = [q for a, q in zip(struct.atoms, charges) if a.element_name == "H"]
    for q in c_charges:
        assert q < 0, f"Expected negative C charge, got {q}"
    for q in h_charges:
        assert q > 0, f"Expected positive H charge, got {q}"


def test_resp_gaff2_benzene(testdir, benzene_sdf):
    wdir = testdir / "gaff2_resp"
    if wdir.is_dir():
        shutil.rmtree(wdir)

    loader = LigandLoader()
    ligands = loader.load(benzene_sdf, only_first=True, name_from_stem=True)
    ligand = ligands[0]

    ff = load_parametrizer("gaff2", "resp", raise_errors=True, keep_cache=True)
    ligand = ff.run(ligand, wdir)

    assert "prmtop" in ligand.auxiliary_files
    assert "inpcrd" in ligand.auxiliary_files
    assert "xml" in ligand.auxiliary_files
    assert "pdb" in ligand.auxiliary_files
    _assert_resp_charges(ligand)

    ligand.dump(wdir)
    assert (wdir / f"{ligand.name}.prmtop").is_file()


def test_resp_openff_benzene(testdir, benzene_sdf):
    if "openff" not in PARAMETRIZER_REGISTRY.names():
        pytest.skip("OpenFF not installed")

    wdir = testdir / "openff_resp"
    if wdir.is_dir():
        shutil.rmtree(wdir)

    loader = LigandLoader()
    ligands = loader.load(benzene_sdf, only_first=True, name_from_stem=True)
    ligand = ligands[0]

    ff = load_parametrizer("openff-2.1.0", "resp", raise_errors=True, keep_cache=True)
    ligand = ff.run(ligand, wdir)

    assert "prmtop" in ligand.auxiliary_files
    assert "inpcrd" in ligand.auxiliary_files
    assert "xml" in ligand.auxiliary_files
    assert "pdb" in ligand.auxiliary_files
    _assert_resp_charges(ligand)

    assert (wdir / f"{ligand.name}.prmtop").is_file()
