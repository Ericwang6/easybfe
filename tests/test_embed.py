import logging
import math
import shutil
from pathlib import Path

import pytest
from rdkit import Chem

from easybfe.docking.embed import constr_embed_with_rdkit

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "_test_embed"

logging.basicConfig(level=logging.DEBUG, format="%(name)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Constrained embedding tests
# ---------------------------------------------------------------------------

def test_constr_embed_with_rdkit():
    if OUT_DIR.is_dir():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()

    suppl = Chem.SDMolSupplier(str(DATA_DIR / "tyk2_ligands.sdf"), removeHs=False)
    mols = [m for m in suppl if m is not None]
    assert len(mols) > 1, "Need at least 2 molecules in the SDF"

    ref_mol = mols[0]
    for mol in mols[1:]:
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown"
        mol.RemoveAllConformers()
        result, mapping = constr_embed_with_rdkit(mol, ref_mol)
        assert result.GetNumConformers() > 0, f"Failed to generate conformer for {mol_name}"
        assert isinstance(mapping, dict) and len(mapping) > 0

        out_path = OUT_DIR / f"{mol_name}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            w.write(result)


# ---------------------------------------------------------------------------
# VinaDocking tests
# ---------------------------------------------------------------------------

vina = pytest.importorskip("vina")

from easybfe.docking.vina import VinaDocking


@pytest.fixture(scope="module")
def tyk2_mols():
    """Load tyk2 ligands; first molecule is used as reference."""
    suppl = Chem.SDMolSupplier(str(DATA_DIR / "tyk2_ligands.sdf"), removeHs=False)
    mols = [m for m in suppl if m is not None]
    assert len(mols) >= 2
    return mols


@pytest.fixture(scope="module")
def vina_docking(tyk2_mols):
    """Create a VinaDocking instance using tyk2 protein and a box
    derived from the first ligand via ref_mol."""
    return VinaDocking(
        protein=DATA_DIR / "tyk2_pdbfixer.pdb",
        ref_mol=tyk2_mols[0],
        protein_prep_exec="obabel",
    )


def test_vina_dock(vina_docking, tyk2_mols):
    mol = tyk2_mols[1]
    poses = vina_docking.dock(mol)
    assert isinstance(poses, list)
    assert len(poses) > 0, "dock() returned no poses"
    for pose in poses:
        assert isinstance(pose, Chem.Mol)
        assert pose.GetNumConformers() > 0


def test_vina_rescore(vina_docking, tyk2_mols):
    mol = tyk2_mols[0]
    score = vina_docking.rescore(mol)
    assert isinstance(score, float)
    assert math.isfinite(score), f"rescore() returned non-finite value: {score}"


def test_vina_constr_dock(vina_docking, tyk2_mols):
    from copy import deepcopy
    mol = deepcopy(tyk2_mols[1])
    ref_mol = tyk2_mols[0]
    result = vina_docking.constr_dock(mol, ref_mol)
    assert isinstance(result, Chem.Mol)
    assert result.GetNumConformers() > 0
    assert result.HasProp('vina_score')
    assert result.HasProp('rmsd')
    assert result.HasProp('ff_energy')
    assert math.isfinite(result.GetDoubleProp('vina_score'))
    assert math.isfinite(result.GetDoubleProp('rmsd'))
    assert math.isfinite(result.GetDoubleProp('ff_energy'))


def test_vina_constr_dock_no_em(vina_docking, tyk2_mols):
    from copy import deepcopy
    mol = deepcopy(tyk2_mols[1])
    ref_mol = tyk2_mols[0]
    result = vina_docking.constr_dock(mol, ref_mol, run_em=False)
    assert isinstance(result, Chem.Mol)
    assert result.GetNumConformers() > 0
    assert result.HasProp('vina_score')
    assert result.HasProp('rmsd')
    assert not result.HasProp('ff_energy')
    assert math.isfinite(result.GetDoubleProp('vina_score'))
    assert math.isfinite(result.GetDoubleProp('rmsd'))
