import os
from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from ..mapping import LazyMCSMapper


def ndarray2quantity(ndarray):
    value = [mm.Vec3(float(arr[0]), float(arr[1]), float(arr[2])) for arr in ndarray]
    quantity = unit.Quantity(value, unit=unit.nanometers)
    return quantity


def constr_embed(mol: Chem.Mol, ref_mol: Chem.Mol, mcs=None, max_attempts=2000, num_confs=20, run_em=False, protein_pdb=None, out_sdf=None, constr=True, restr: float = 1.0):
    mol = Chem.AddHs(mol)
    atom_mapping = LazyMCSMapper(mcs, use_positions=False).run_mapping(mol, ref_mol)
    coord_map = {}
    for k, v in atom_mapping.items():
        coord_map[k] = ref_mol.GetConformer().GetAtomPosition(v)
    
    # ci = AllChem.EmbedMolecule(mol, coordMap=coord_map, enforceChirality=True, maxAttempts=max_attempts)
    ci = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, coordMap=coord_map, enforceChirality=True, maxAttempts=max_attempts, clearConfs=True)
    assert mol.GetNumConformers() > 0, "Could not embed molecule"

    for i in range(mol.GetNumConformers()):
        rms = rdMolAlign.AlignMol(mol, ref_mol, prbCid=i, atomMap=list(atom_mapping.items()))
    
    if run_em:
        # Manual Constrained
        for conf in mol.GetConformers():
            for k in coord_map:
                conf.SetAtomPosition(k, coord_map[k])
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        off_mol.assign_partial_charges('gasteiger')
        generator = SMIRNOFFTemplateGenerator(molecules=[off_mol]).generator
        ligand_top = Topology.from_molecules(off_mol).to_openmm()
        ligand_pos = mol.GetConformer(0).GetPositions() / 10
        num_ligand_atoms = ligand_top.getNumAtoms()

        if protein_pdb:
            pdb = app.PDBFile(protein_pdb)
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.add(ligand_top, ndarray2quantity(ligand_pos))
            top, pos = modeller.getTopology(), modeller.getPositions()
            ff = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
            pos = np.array([[vec.x, vec.y, vec.z] for vec in pos])
        else:
            top, pos = ligand_top, ligand_pos
            ff = app.ForceField()
        
        ff.registerTemplateGenerator(generator)
        system = ff.createSystem(
            top, 
            nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, 
            constraints=None, rigidWater=False, removeCMMotion=False
        )
        context = mm.Context(
            system, 
            mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 2*unit.femtoseconds)
        )

        # Constrained all protein atoms
        residues = list(top.residues())
        for residue in residues[:-1]:
            for atom in residue.atoms():
                system.setParticleMass(atom.index, 0.0 * unit.amu)
        
        # Find constrained indices
        constr_indices = []
        ligand_residue = list(top.residues())[-1]
        for i, atom in enumerate(ligand_residue.atoms()):
            if i in atom_mapping:
                constr_indices.append(atom.index)

        if constr:
            for index in constr_indices:
                system.setParticleMass(atom.index, 0.0 * unit.amu)
        else:
            # Use restrained, not constrained
            force_constant = restr * unit.kilocalories_per_mole/unit.angstroms**2
            restraint_force = mm.CustomExternalForce('(k/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
            restraint_force.addGlobalParameter('k', force_constant)
            restraint_force.addPerParticleParameter('x0')
            restraint_force.addPerParticleParameter('y0')
            restraint_force.addPerParticleParameter('z0')
            for index in constr_indices:
                restraint_force.addParticle(index, pos[index])
            system.addForce(restraint_force)

        energies = []
        for ci in range(mol.GetNumConformers()):
            pos[-num_ligand_atoms:] = mol.GetConformer(ci).GetPositions() / 10
            context.setPositions(pos)
            mm.LocalEnergyMinimizer.minimize(context)
            state = context.getState(getPositions=True, getEnergy=True)
            min_positions = state.getPositions(asNumpy=True)._value
            energy = float(state.getPotentialEnergy()._value)
            energies.append(energy)
            for i, vec in enumerate(min_positions[-num_ligand_atoms:]):
                mol.GetConformer(ci).SetAtomPosition(i, [float(x) * 10 for x in vec])
    
    if out_sdf:
        with Chem.SDWriter(out_sdf) as w:
            for ci in np.argsort(energies):
                mol.SetDoubleProp("Energy", energies[ci])
                w.write(mol, confId=int(ci))
    return mol


    
def constrained_optimization(mol: Chem.Mol, protein_pdb: os.PathLike, coord_map: Dict[int, np.ndarray], constr: bool = True, restr: float = 10.0):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    off_mol.assign_partial_charges('gasteiger')
    generator = SMIRNOFFTemplateGenerator(molecules=[off_mol]).generator
    ligand_top = Topology.from_molecules(off_mol).to_openmm()
    ligand_pos = mol.GetConformer(0).GetPositions() / 10
    num_ligand_atoms = ligand_top.getNumAtoms()

    # Ligand only, no protein
    ligand_ff = app.ForceField()
    ligand_ff.registerTemplateGenerator(generator)
    system = ligand_ff.createSystem(
        ligand_top, 
        nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, 
        constraints=None, rigidWater=False, removeCMMotion=False
    )

    constrIndices = []
    for atom in ligand_top.atoms():
        if atom.index in coord_map and atom.element.symbol != 'H':
            constrIndices.append(atom.index)
    if constr:
        for index in constrIndices:
            system.setParticleMass(index, 0.0 * unit.amu)
    else:
        # Use restrained, not constrained
        force_constant = restr * unit.kilocalories_per_mole/unit.angstroms**2
        restraint_force = mm.CustomExternalForce('(k/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
        restraint_force.addGlobalParameter('k', force_constant)
        restraint_force.addPerParticleParameter('x0')
        restraint_force.addPerParticleParameter('y0')
        restraint_force.addPerParticleParameter('z0')
        for index in constrIndices:
            restraint_force.addParticle(index, coord_map[index] / 10)
        system.addForce(restraint_force)

    context = mm.Context(
        system, 
        mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 2*unit.femtoseconds)
    )

    energies = []
    for ci in range(mol.GetNumConformers()):
        pos = mol.GetConformer(ci).GetPositions() / 10
        for index in coord_map:
            pos[index] = coord_map[index] / 10
        context.setPositions(pos)
        mm.LocalEnergyMinimizer.minimize(context)
        state = context.getState(getPositions=True, getEnergy=True)
        min_positions = state.getPositions(asNumpy=True)._value
        energy = float(state.getPotentialEnergy()._value)
        energies.append(energy)
        for i, vec in enumerate(min_positions):
            mol.GetConformer(ci).SetAtomPosition(i, [float(x) * 10 for x in vec])
    
    # no protein, return
    if not protein_pdb:
        return mol, energies
    
    # minimization with protein
    pdb = app.PDBFile(str(protein_pdb))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    ligand_pos = mol.GetConformer(0).GetPositions() / 10
    modeller.add(ligand_top, ndarray2quantity(ligand_pos))
    top, pos = modeller.getTopology(), modeller.getPositions()
    ff = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    pos = np.array([[vec.x, vec.y, vec.z] for vec in pos])
    ff.registerTemplateGenerator(generator)
    system = ff.createSystem(
        top, 
        nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, 
        constraints=None, rigidWater=False, removeCMMotion=False
    )
    context = mm.Context(
        system, 
        mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 2*unit.femtoseconds)
    )

    constrIndices = []
    # protein heavy atoms
    residues = list(top.residues())
    for residue in residues[:-1]:
        for atom in residue.atoms():
            if atom.element.symbol != 'H':
                constrIndices.append(atom.index)
    # ligand heavt atoms
    ligand_residue = list(top.residues())[-1]
    for i, atom in enumerate(ligand_residue.atoms()):
        if i in coord_map and atom.element.symbol != 'H':
            constrIndices.append(atom.index)

    if constr:
        for index in constrIndices:
            system.setParticleMass(index, 0.0 * unit.amu)
    else:
        # Use restrained, not constrained
        force_constant = restr * unit.kilocalories_per_mole/unit.angstroms**2
        restraint_force = mm.CustomExternalForce('(k/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
        restraint_force.addGlobalParameter('k', force_constant)
        restraint_force.addPerParticleParameter('x0')
        restraint_force.addPerParticleParameter('y0')
        restraint_force.addPerParticleParameter('z0')
        for index in constrIndices:
            restraint_force.addParticle(index, pos[index])
        system.addForce(restraint_force)

    energies = []
    for ci in range(mol.GetNumConformers()):
        pos[-num_ligand_atoms:] = mol.GetConformer(ci).GetPositions() / 10
        context.setPositions(pos)
        mm.LocalEnergyMinimizer.minimize(context)
        state = context.getState(getPositions=True, getEnergy=True)
        min_positions = state.getPositions(asNumpy=True)._value
        energy = float(state.getPotentialEnergy()._value)
        energies.append(energy)
        for i, vec in enumerate(min_positions[-num_ligand_atoms:]):
            mol.GetConformer(ci).SetAtomPosition(i, [float(x) * 10 for x in vec])
    
    return mol, energies