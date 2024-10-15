import os
from pathlib import Path
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from easybfe.smff import GAFF, OpenFF


@pytest.mark.parametrize(
    "atype, charge_method",
    [('gaff', 'bcc'), ('gaff', 'gas'), ('gaff2', 'bcc'), ('gaff2', 'gas')]
)
def test_gaff(atype, charge_method):
    mol = Chem.AddHs(Chem.MolFromSmiles('C'))
    AllChem.EmbedMolecule(mol)
    testdir = Path(__file__).parent / '_test_smff'
    testdir.mkdir(exist_ok=True)
    fpath = testdir / 'methane.sdf'
    with Chem.SDWriter(str(fpath)) as w:
        w.write(mol)
    
    wdir = testdir / f'{atype}_{charge_method}'
    wdir.mkdir(exist_ok=True)
    smff = GAFF(atype, charge_method)
    smff.parametrize(fpath, wdir)
    assert Path.is_file(wdir / (fpath.stem + '.prmtop'))
    assert Path.is_file(wdir / (fpath.stem + '.top'))


@pytest.mark.parametrize(
    "ff, charge_method",
    [('openff-2.1.0', 'bcc'), ('openff-2.1.0', 'gas')]
)
def test_openff(ff, charge_method):
    mol = Chem.AddHs(Chem.MolFromSmiles('CO'))
    AllChem.EmbedMolecule(mol)
    testdir = Path(__file__).parent / '_test_smff'
    testdir.mkdir(exist_ok=True)
    fpath = testdir / 'methanol.sdf'
    with Chem.SDWriter(str(fpath)) as w:
        w.write(mol)
    
    wdir = testdir / f'{ff}_{charge_method}'
    wdir.mkdir(exist_ok=True)
    smff = OpenFF(ff, charge_method)
    smff.parametrize(fpath, wdir)
    assert Path.is_file(wdir / (fpath.stem + '.prmtop'))
    assert Path.is_file(wdir / (fpath.stem + '.top'))
