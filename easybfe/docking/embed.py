import os
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from ..core import Ligand
from ..mapping import LazyMCSMapper, LomapAtomMapper

logger = logging.getLogger(__name__)


def _ndarray_to_quantity(ndarray: np.ndarray):
    """Convert an ``(N, 3)`` numpy array (in nm) to an OpenMM ``Quantity``."""
    value = [mm.Vec3(float(arr[0]), float(arr[1]), float(arr[2])) for arr in ndarray]
    return unit.Quantity(value, unit=unit.nanometers)


def constr_embed_with_rdkit(
    mol: Chem.Mol,
    ref_mol: Chem.Mol,
    mapping: Optional[dict[int, int]] = None,
    max_attempts: int = 10000,
    max_displ: float = 0.1,
    force_constant: float = 1000.0,
) -> Tuple[Chem.Mol, dict[int, int]]:
    """Embed *mol* with coordinates constrained to match *ref_mol*.

    Generates a 3-D conformer for *mol* where mapped heavy atoms are
    positioned at the corresponding coordinates of *ref_mol*, followed
    by an MMFF94 minimisation with position constraints.

    Parameters
    ----------
    mol : :class:`rdkit.Chem.Mol`
        Probe molecule (conformers will be cleared before embedding).
    ref_mol : :class:`rdkit.Chem.Mol`
        Reference molecule with a 3-D conformer.
    mapping : dict[int, int], optional
        Atom mapping ``{mol_idx: ref_mol_idx}``.  When *None* a mapping
        is generated automatically via :class:`~easybfe.mapping.LomapAtomMapper`.
    max_attempts : int
        Maximum RDKit embedding attempts.
    max_displ : float
        Maximum displacement (Angstrom) for MMFF position constraints.
    force_constant : float
        Force constant (kcal/mol/A^2) for MMFF position constraints.

    Returns
    -------
    mol : :class:`rdkit.Chem.Mol`
        Molecule with a new conformer.
    mapping : dict[int, int]
        The atom mapping used (either the one provided or the auto-generated one).
    """
    mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'probe'
    ref_name = ref_mol.GetProp('_Name') if ref_mol.HasProp('_Name') else 'ref'

    if mapping is None:
        logger.info("%s -> %s: computing atom mapping via LomapAtomMapper", mol_name, ref_name)
        mapper = LomapAtomMapper(max3d=1000.0, threed=False)
        lig_a = Ligand(
            name=mol_name,
            smiles=Chem.MolToSmiles(mol),
            mol_block=Chem.MolToMolBlock(mol),
        )
        lig_b = Ligand(
            name=ref_name,
            smiles=Chem.MolToSmiles(ref_mol),
            mol_block=Chem.MolToMolBlock(ref_mol),
        )
        raw_mapping = mapper.propose_mapping(lig_a, lig_b)
        mapping = mapper.post_process_mapping(
            lig_a.get_rdmol(), lig_b.get_rdmol(), raw_mapping
        )
    else:
        logger.info("%s -> %s: using provided atom mapping (%d pairs)", mol_name, ref_name, len(mapping))

    heavy_mapped = {k: v for k, v in mapping.items()
                    if mol.GetAtomWithIdx(k).GetAtomicNum() != 1}
    logger.info("%s -> %s: mapping has %d mapped atoms (%d heavy atom, %d hydrogen)",
                mol_name, ref_name, len(mapping), len(heavy_mapped), len(mapping) - len(heavy_mapped))

    coord_map = {}
    for k, v in heavy_mapped.items():
        p = ref_mol.GetConformer().GetAtomPosition(v)
        coord_map[k] = p
    ci = AllChem.EmbedMolecule(
        mol, coordMap=coord_map, enforceChirality=True, maxAttempts=max_attempts, useRandomCoords=True
    )
    if ci < 0:
        logger.warning("%s: RDKit constraint embedding failed, falling back to unconstrained + align",
                       mol_name)
        ci = AllChem.EmbedMolecule(mol, enforceChirality=True, maxAttempts=max_attempts,
            useRandomCoords=True,
        )
        assert mol.GetNumConformers() > 0, f"Could not embed molecule {mol_name}"
        rdMolAlign.AlignMol(mol, ref_mol, atomMap=list(mapping.items()))
        conf = mol.GetConformer(0)
        for k, pos in coord_map.items():
            conf.SetAtomPosition(k, pos)
    else:
        logger.debug("%s: RDKit constraint embedding succeeded", mol_name)

    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
    if mp is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=0)
        for atom_idx in heavy_mapped:
            ff.MMFFAddPositionConstraint(atom_idx, max_displ, force_constant)
        converged = ff.Minimize(maxIts=2000)
        energy = ff.CalcEnergy()
        logger.info("%s: MMFF94 minimization %s (energy=%.2f kcal/mol)",
                    mol_name, "converged" if converged == 0 else "did not converge", energy)
    else:
        logger.warning("%s: could not obtain MMFF94 properties, skipping minimization", mol_name)

    return mol, mapping


def constrained_em_with_protein(
    mol: Chem.Mol,
    protein_pdb: os.PathLike,
    coord_map: Dict[int, np.ndarray],
    constrain: bool = True,
    restraint_k: float = 10.0,
) -> Tuple[Chem.Mol, float]:
    """Energy-minimise a ligand in the presence of a protein receptor.

    All protein heavy atoms are frozen.  Mapped ligand heavy atoms are
    either frozen (``constrain=True``) or harmonically restrained
    (``constrain=False``) to positions given in *coord_map*.

    Parameters
    ----------
    mol : :class:`rdkit.Chem.Mol`
        Ligand with exactly one conformer (positions in Angstrom).
    protein_pdb : os.PathLike
        Path to the protein PDB file.
    coord_map : dict[int, numpy.ndarray]
        ``{ligand_atom_idx: target_position}`` where positions are in
        Angstrom.  Only heavy-atom entries are used.
    constrain : bool
        If *True* freeze mapped heavy atoms (mass = 0).  If *False*
        apply harmonic position restraints with *restraint_k*.
    restraint_k : float
        Harmonic restraint force constant in kcal/(mol * A^2).
        Only used when ``constrain=False``.

    Returns
    -------
    mol : :class:`rdkit.Chem.Mol`
        Ligand with updated coordinates from minimisation.
    energy : float
        Potential energy after minimisation (kJ/mol).
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    off_mol.assign_partial_charges('gasteiger')
    generator = SMIRNOFFTemplateGenerator(molecules=[off_mol]).generator
    ligand_top = Topology.from_molecules(off_mol).to_openmm()
    ligand_pos = mol.GetConformer(0).GetPositions() / 10  # Angstrom -> nm
    num_ligand_atoms = ligand_top.getNumAtoms()

    ff = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    ff.registerTemplateGenerator(generator)

    pdb = app.PDBFile(str(protein_pdb))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    modeller.add(ligand_top, _ndarray_to_quantity(ligand_pos))
    top = modeller.getTopology()
    pos = np.array([[vec.x, vec.y, vec.z] for vec in modeller.getPositions()])

    system = ff.createSystem(
        top,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )

    frozen_indices = []
    # Freeze all protein heavy atoms
    residues = list(top.residues())
    for residue in residues[:-1]:
        for atom in residue.atoms():
            if atom.element.symbol != 'H':
                frozen_indices.append(atom.index)
    # Mapped ligand heavy atoms
    ligand_residue = residues[-1]
    mapped_ligand_global = []
    global_to_target_nm = {}
    for i, atom in enumerate(ligand_residue.atoms()):
        if i in coord_map and atom.element.symbol != 'H':
            mapped_ligand_global.append(atom.index)
            global_to_target_nm[atom.index] = np.asarray(coord_map[i], dtype=float) / 10

    if constrain:
        for idx in frozen_indices + mapped_ligand_global:
            system.setParticleMass(idx, 0.0 * unit.amu)
    else:
        for idx in frozen_indices:
            system.setParticleMass(idx, 0.0 * unit.amu)
        k_unit = restraint_k * unit.kilocalories_per_mole / unit.angstroms ** 2
        restraint_force = mm.CustomExternalForce(
            '(k/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)'
        )
        restraint_force.addGlobalParameter('k', k_unit)
        restraint_force.addPerParticleParameter('x0')
        restraint_force.addPerParticleParameter('y0')
        restraint_force.addPerParticleParameter('z0')
        for idx in mapped_ligand_global:
            restraint_force.addParticle(idx, global_to_target_nm[idx])
        system.addForce(restraint_force)

    context = mm.Context(
        system,
        mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picoseconds, 2 * unit.femtoseconds),
    )

    pos[-num_ligand_atoms:] = mol.GetConformer(0).GetPositions() / 10
    for global_idx, target_nm in global_to_target_nm.items():
        pos[global_idx] = target_nm
    context.setPositions(pos)
    mm.LocalEnergyMinimizer.minimize(context)
    state = context.getState(getPositions=True, getEnergy=True)
    min_positions = state.getPositions(asNumpy=True)._value
    energy = float(state.getPotentialEnergy()._value)

    conf = mol.GetConformer(0)
    for i, vec in enumerate(min_positions[-num_ligand_atoms:]):
        conf.SetAtomPosition(i, [float(x) * 10 for x in vec])

    logger.info("OpenMM EM done (energy=%.2f kJ/mol)", energy)
    return mol, energy
