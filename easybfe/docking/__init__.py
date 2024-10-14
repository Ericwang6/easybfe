import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import tempfile
from ..smff import OpenFF
from ..mapping import LazyMCSMapper
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator


def ndarray2quantity(ndarray):
    value = [mm.Vec3(float(arr[0]), float(arr[1]), float(arr[2])) for arr in ndarray]
    quantity = unit.Quantity(value, unit=unit.nanometers)
    return quantity


def constr_embed(mol: Chem.Mol, ref_mol: Chem.Mol, mcs=None, max_attempts=2000, num_confs=20, run_em=False, protein_pdb=None, out_sdf=None):
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
    
    # Manual Constrained
    for conf in mol.GetConformers():
        for k in coord_map:
            conf.SetAtomPosition(k, coord_map[k])
    
    if run_em:
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
        system = ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
        context = mm.Context(system, mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 2*unit.femtoseconds))

        ligand_residue = list(top.residues())[-1]
        for i, atom in enumerate(ligand_residue.atoms()):
            if i in atom_mapping:
                system.setParticleMass(atom.index, 0.0 * unit.amu)
        
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


    
