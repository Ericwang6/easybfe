import pytest
from pathlib import Path
import numpy as np
import openmm.app as app
import openmm.unit as unit
import parmed


# OpenMM creates a tip3p water with 2-bond and 1-angle
# But in AMBER, if you need to use SETTLE to treat TIP3P, you need the water to have 3-bonds and 1 angle
# Otherwise pmemd.cuda will raise an error

def test_modify_openmm_water():
    wdir = Path(__file__).resolve().parent / 'data'
    pdb = app.PDBFile(str(wdir / 'water.pdb'))
    ff = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    system = ff.createSystem(pdb.topology, rigidWater=False)
    struct = parmed.openmm.load_topology(pdb.topology, system)
    
    for residue in struct.residues:
        # Amber uses WAT to identify SETTLE for waters
        if residue.name == 'HOH':
            residue.name = 'WAT'
        if residue.name != 'HOH' and residue.name != 'WAT':
            continue
        atom_dict = {atom.name: atom for atom in residue.atoms}
        hhtype = parmed.BondType(k=553.000, req=1.514, list=struct.bond_types)
        struct.bond_types.append(hhtype)
        hh_bond = parmed.Bond(atom_dict['H1'], atom_dict['H2'], hhtype)
        struct.bonds.append(hh_bond)
    
    to_del = []
    for i, angle in enumerate(struct.angles):
        resname = angle.atom1.residue.name
        if resname == 'HOH' or resname == 'WAT':
            to_del.append(i)
    for item in reversed(to_del):
        struct.angles.pop(item)

    struct.save(str(wdir / '_test_water.prmtop'), overwrite=True)

    struct_read = parmed.load_file(str(wdir / '_test_water.prmtop'))
    assert len(struct_read.angles) == 0
    for bond in struct_read.bonds:
        if bond.atom1.name == 'H1' and bond.atom2.name == 'H2':
            assert np.allclose(bond.type.k, 553.000)
            assert np.allclose(bond.type.req, 1.514)
        else:
            assert np.allclose(bond.type.req, 0.9572)
        

