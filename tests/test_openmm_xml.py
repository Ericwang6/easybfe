import pytest
import numpy as np
from pathlib import Path
from collections import defaultdict
from easybfe.smff import GAFF, load_parametrizer, OpenFF
from easybfe.smff.utils import convert_to_xml
from easybfe.core.ligand import LigandLoader

from rdkit import Chem
import openmm as mm
import openmm.app as app
import openmm.unit as unit


def getEnergyDecomposition(system, positions):
    integrator = mm.VerletIntegrator(0.1)
    context = mm.Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)  # Get the state with total energy

    ene = {}
    # Total potential energy of the system
    total_energy = state.getPotentialEnergy()

    ene['total'] = total_energy.value_in_unit(unit.kilojoule_per_mole)

    # Loop over all forces in the system
    for i, force in enumerate(system.getForces()):
        # Disable all forces except the one we're interested in
        for j, f in enumerate(system.getForces()):
            f.setForceGroup(j)
        
        # Get energy for this specific force
        context.reinitialize(preserveState=True)
        state = context.getState(getEnergy=True, groups={i})
        force_energy = state.getPotentialEnergy()

        # Print the energy contribution of this force
        force_name = force.__class__.__name__
        if force_name == 'CMMotionRemover':
            continue
        ene[force_name] = force_energy.value_in_unit(unit.kilojoule_per_mole)
    return ene


def diagnose_torsion(system: mm.System, top: app.Topology):
    atoms = list(top.atoms())
    bondedInfo = defaultdict(list)
    for bond in top.bonds():
        bondedInfo[bond[0]].append(bond[1])
        bondedInfo[bond[1]].append(bond[0])
    
    torsionForces = [f for f in system.getForces() if isinstance(f, mm.PeriodicTorsionForce)]
    propers, impropers = [], []
    
    for torsionForce in torsionForces:
        for i in range(torsionForce.getNumTorsions()):
            param = torsionForce.getTorsionParameters(i)
            atom1, atom2, atom3, atom4 = tuple([atoms[idx] for idx in param[:4]])
            if param[-1].value_in_unit(unit.kilojoules_per_mole) == 0:
                continue
            if (atom2 in bondedInfo[atom1]) and (atom3 in bondedInfo[atom2]) and (atom4 in bondedInfo[atom3]):
                propers.append(param)
            else:
                impropers.append(param)
    print(len(propers), len(impropers))
    return propers, impropers


def test_openmm_xml_openff():
    if OpenFF is None:
        pytest.skip("OpenFF not installed")
    wdir = Path(__file__).parent / '_test_openmm_xml_openff'
    wdir.mkdir(exist_ok=True)
    sdf = Path(__file__).parent / 'data/CDD_1819.sdf'
    mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
    pos = mol.GetConformer().GetPositions() / 10

    loader = LigandLoader()
    ligands = loader.load(sdf, only_first=True, name_from_stem=True)
    assert len(ligands) == 1
    ff = load_parametrizer('openff-2.2.0', 'gas', engine='openff')
    ligand = ff.run(ligands[0], nprocs=1)
    ligand.dump(wdir)

    f_prmtop = str(wdir / 'CDD_1819.prmtop')
    prmtop = app.AmberPrmtopFile(f_prmtop)
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    ene_direct = getEnergyDecomposition(system, pos)
    diagnose_torsion(system, prmtop.topology)

    ene_read = getEnergyDecomposition(system, pos)
    assert np.allclose(ene_direct['total'], ene_read['total'])

    f_xml = str(wdir / 'CDD_1819.xml')
    convert_to_xml(f_prmtop, f_xml)
    forcefield = app.ForceField(f_xml)
    system = forcefield.createSystem(
        prmtop.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None, rigidWater=False
    )
    ene_convert = getEnergyDecomposition(system, pos)
    diagnose_torsion(system, prmtop.topology)
    assert np.allclose(ene_direct['total'], ene_convert['total'])


def test_openmm_xml_gaff():
    wdir = Path(__file__).parent / '_test_openmm_xml_gaff'
    wdir.mkdir(exist_ok=True)
    sdf = Path(__file__).parent / 'data/CDD_1819.sdf'
    mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
    pos = mol.GetConformer().GetPositions() / 10

    loader = LigandLoader()
    ligands = loader.load(sdf, only_first=True, name_from_stem=True)
    assert len(ligands) == 1
    ff = GAFF(forcefield='gaff2', charge_method='gas')
    ligand = ff.run(ligands[0], nprocs=1)
    ligand.dump(wdir)

    f_prmtop = str(wdir / 'CDD_1819.prmtop')
    prmtop = app.AmberPrmtopFile(f_prmtop)
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    ene_read = getEnergyDecomposition(system, pos)
    diagnose_torsion(system, prmtop.topology)

    # Read convert
    f_xml = str(wdir / 'CDD_1819.xml')
    convert_to_xml(f_prmtop, f_xml)
    forcefield = app.ForceField(f_xml)
    system = forcefield.createSystem(
        prmtop.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None, rigidWater=False
    )
    ene_convert = getEnergyDecomposition(system, pos)
    diagnose_torsion(system, prmtop.topology)
    assert np.allclose(ene_read['total'], ene_convert['total'])


def test_join_dihedrals():
    prmtop = Path(__file__).parent / 'data/enamine_19451.prmtop'
    convert_to_xml(str(prmtop), prmtop.with_name('enamine_19451.xml'))