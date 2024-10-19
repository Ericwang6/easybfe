import os
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple, Any

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parmed
from rdkit import Chem

from ..smff.utils import convert_to_xml


PROTEIN_FF_XMLS = {
    'ff14SB': 'amber14/protein.ff14SB.xml'
}

WATER_FF_XMLS = {
    'tip3p': 'amber14/tip3p.xml'
}


def _check_renumber_mapping_dict(mapping_dict: Dict[int, int], natoms: int):
    ref_list = list(range(natoms))
    old_numbers = list(mapping_dict.keys())
    old_numbers.sort()
    assert ref_list == old_numbers, f"Old numbers not valid"
    new_numbers = list(mapping_dict.values())
    new_numbers.sort()
    assert ref_list == new_numbers, f"New numbers not valid"


def renumber_rdkit_mol(old_mol: Chem.Mol, mapping_dict: Dict[int, int]):
    _check_renumber_mapping_dict(mapping_dict, old_mol.GetNumAtoms())
    new_mol = Chem.RWMol()
    # new mapping -> old mapping
    rev_dict = {v:k for k,v in mapping_dict.items()}
    for k in range(old_mol.GetNumAtoms()):
        atom = old_mol.GetAtomWithIdx(rev_dict[k])
        new_mol.AddAtom(atom)
    
    for bond in old_mol.GetBonds():
        new_begin_idx = mapping_dict[bond.GetBeginAtomIdx()]
        new_end_idx = mapping_dict[bond.GetEndAtomIdx()]
        new_mol.AddBond(new_begin_idx, new_end_idx, bond.GetBondType())
    conf = old_mol.GetConformer()
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())
    for old_idx, new_idx in mapping_dict.items():
        pos = conf.GetAtomPosition(old_idx)
        new_conf.SetAtomPosition(new_idx, pos)
    new_mol.AddConformer(new_conf)
    new_mol = new_mol.GetMol()
    new_mol.SetProp('_Name', old_mol.GetProp('_Name'))
    return new_mol


def mk_renumber_mapping_from_cc(cc_list: List[int], natoms: int):
    sc_offset = len(cc_list)
    cc_offset = 0
    mapping = {}
    for i in range(natoms):
        if i in cc_list:
            mapping[i] = cc_offset
            cc_offset += 1
        else:
            mapping[i] = sc_offset
            sc_offset += 1
    return mapping


def renumber_parmed_structure(struct: parmed.Structure, mapping: Dict[int, int]):
    _check_renumber_mapping_dict(mapping, len(struct.atoms))
    rev_mapping = {v: k for k, v in mapping.items()}
    new_atoms = parmed.AtomList()
    for i in range(len(struct.atoms)):
        new_atoms.append(struct.atoms[rev_mapping[i]])
    new_atoms.claim()
    struct.atoms = new_atoms
    return struct


def computeBoxVectorsWithPadding(positions: unit.Quantity, buffer: unit.Quantity):
    min_x, min_y, min_z = min(pos.x for pos in positions), min(pos.y for pos in positions), min(pos.z for pos in positions)
    max_x, max_y, max_z = max(pos.x for pos in positions), max(pos.y for pos in positions), max(pos.z for pos in positions)
    buffer = buffer.value_in_unit(unit.nanometer)
    box_x = (max_x - min_x) + buffer * 2
    box_y = (max_y - min_y) + buffer * 2
    box_z = (max_z - min_z) + buffer * 2
    return (
        mm.Vec3(box_x, 0.0, 0.0),
        mm.Vec3(0.0, box_y, 0.0),
        mm.Vec3(0.0, 0.0, box_z)
    )


def shiftToBoxCenter(positions: unit.Quantity, boxVectors: Tuple[mm.Vec3]):
    positions = positions.value_in_unit(unit.nanometers)
    boxCenter = (boxVectors[0] + boxVectors[1] + boxVectors[2]) / 2
    minVec = mm.Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
    maxVec = mm.Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
    posCenter = (minVec + maxVec) * 0.5
    shiftVec = boxCenter - posCenter
    newPositions = unit.Quantity(value=[pos + shiftVec for pos in positions], unit=unit.nanometers)
    return newPositions


def generate_amber_mask(natomsA: int, natomsB: int, mapping: Dict[int, int]):
    scA = [i for i in range(natomsA) if i not in mapping.keys()]
    scB = [i for i in range(natomsB) if i not in mapping.values()]
    res = {
        "noshakemask": f"'@1-{natomsA+natomsB}'",
        "timask1": f"'@1-{natomsA}'",
        "timask2": f"'@{natomsA+1}-{natomsA+natomsB}'",
        "scmask1": "'@{}'".format(','.join(str(i+1) for i in scA)),
        "scmask2": "'@{}'".format(','.join(str(i+1+natomsA) for i in scB))
    }
    return res


def hydrogen_mass_repartition(struct: parmed.Structure, hydrogen_mass: float = 3.024, dowater: bool = False):
    for residue in struct.residues:
        if (not dowater) and (residue.name in ['WAT', 'HOH']):
            continue
        for atom in residue.atoms:
            if atom.element != 1:
                subtract_mass = 0
                for nei in atom.bond_partners:
                    if nei.element == 1:
                        subtract_mass += hydrogen_mass - nei.mass
                        nei.mass = hydrogen_mass
                atom.mass -= subtract_mass


def prep_ligand_rbfe_systems(
    protein_pdb: os.PathLike,
    ligandA_mol: Union[os.PathLike, Chem.Mol],
    ligandA_top: os.PathLike,
    ligandB_mol: Union[os.PathLike, Chem.Mol],
    ligandB_top: os.PathLike,
    mapping: Dict[int, int],
    wdir: os.PathLike,
    protein_ff: str = 'ff14SB',
    water_ff: str = 'tip3p',
    gas_config: Dict[str, Any] = dict(),
    solvent_config: Dict[str, Any] = dict(),
    complex_config: Dict[str, Any] = dict()
):
    # Create working directory
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)

    if not isinstance(ligandA_mol, Chem.Mol):
        ligandA_mol = Chem.SDMolSupplier(ligandA_mol, removeHs=False)[0]
    if not isinstance(ligandB_mol, Chem.Mol):
        ligandB_mol = Chem.SDMolSupplier(ligandB_mol, removeHs=False)[0]
    
    ligandA_struct = parmed.load_file(str(ligandA_top))
    ligandA_struct.residues[0].name = 'MOLA'
    ligandA_struct.residues[0].resname = 'MOLA'
    ligandA_struct.coordinates = ligandA_mol.GetConformer().GetPositions()

    ligandB_struct = parmed.load_file(str(ligandB_top))
    ligandB_struct.residues[0].name = 'MOLB'
    ligandB_struct.coordinates = ligandB_mol.GetConformer().GetPositions()

    # Make renumbered topologies based on atom mapping
    # This make the order of common core atoms the same, which is a requirement of Amber hybrid topologies
    ccA, ccB = [], []
    for k, v in mapping.items():
        ccA.append(k)
        ccB.append(v)
    renum_map_A = mk_renumber_mapping_from_cc(ccA, ligandA_mol.GetNumAtoms())
    renum_map_B = mk_renumber_mapping_from_cc(ccB, ligandB_mol.GetNumAtoms())

    with Chem.SDWriter(str(wdir / 'ligandA_renum.sdf')) as w:
        ligandA_renum = renumber_rdkit_mol(ligandA_mol, renum_map_A)
        w.write(ligandA_renum)
    with Chem.SDWriter(str(wdir / 'ligandB_renum.sdf')) as w:
        ligandB_renum = renumber_rdkit_mol(ligandB_mol, renum_map_B)
        w.write(ligandB_renum)

    # Write amber mask
    with open(wdir / 'mask.json', 'w') as fp:
        json.dump(
            generate_amber_mask(ligandA_mol.GetNumAtoms(), ligandB_mol.GetNumAtoms(), {i:i for i in range(len(mapping))}),
            fp,
            indent=4
        )

    renumber_parmed_structure(ligandA_struct, renum_map_A)
    renumber_parmed_structure(ligandB_struct, renum_map_B)

    ffxml_A = str(wdir / 'ligandA.xml')
    convert_to_xml(ligandA_struct, ffxml_A)
    ffxml_B = str(wdir / 'ligandB.xml')
    convert_to_xml(ligandB_struct, ffxml_B)

    if protein_ff.endswith('.xml'):
        protein_ff_xml = protein_ff
    else:
        protein_ff_xml = PROTEIN_FF_XMLS[protein_ff]
    
    if water_ff.endswith('.xml'):
        water_ff_xml = water_ff
    else:
        water_ff_xml = WATER_FF_XMLS[water_ff]
    
    ff = app.ForceField(protein_ff_xml, water_ff_xml, ffxml_A, ffxml_B)

    def _save(modeller, basename, do_hmr=True, hydrogen_mass=3.024, do_hmr_water=False, **kwargs):
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False)
        tmp = parmed.openmm.load_topology(modeller.topology, system, xyz=modeller.positions)

        # Hydrogen mass repartition
        if do_hmr:
            hydrogen_mass_repartition(tmp, hydrogen_mass, do_hmr_water)
        
        # Note: In AmberTools, TIP3P water will have 3 bonds, 0 angles. If not, pmemd will raise an error:
        # `Error: Fast 3-point water residue, name and bond data incorrect!` 
        for residue in tmp.residues:
            # Amber uses WAT to identify SETTLE for waters
            if residue.name == 'HOH':
                residue.name = 'WAT'
            if residue.name != 'HOH' and residue.name != 'WAT':
                continue
            atom_dict = {atom.name: atom for atom in residue.atoms}
            hhtype = parmed.BondType(k=553.000, req=1.514, list=tmp.bond_types)
            tmp.bond_types.append(hhtype)
            hh_bond = parmed.Bond(atom_dict['H1'], atom_dict['H2'], hhtype)
            tmp.bonds.append(hh_bond)
        
        to_del = []
        for i, angle in enumerate(tmp.angles):
            resname = angle.atom1.residue.name
            if resname == 'HOH' or resname == 'WAT':
                to_del.append(i)
        for item in reversed(to_del):
            tmp.angles.pop(item)

        tmp.save(str(wdir / f'{basename}.prmtop'), overwrite=True)
        tmp.save(str(wdir / f'{basename}.inpcrd'), overwrite=True)
        tmp.save(str(wdir / f'{basename}.pdb'), overwrite=True)

    modeller = app.Modeller(app.Topology(), [])
    modeller.add(ligandA_struct.topology, ligandA_struct.positions)
    modeller.add(ligandB_struct.topology, ligandB_struct.positions)
    
    # Gas-Phase
    gas_buffer = gas_config.get('buffer', 20.0) / 10 * unit.nanometers
    gasBoxVectors = computeBoxVectorsWithPadding(modeller.positions, gas_buffer)
    modeller.positions = shiftToBoxCenter(modeller.positions, gasBoxVectors)
    modeller.topology.setPeriodicBoxVectors(gasBoxVectors)
    _save(modeller, 'gas', **gas_config)
    
    # Solvent phase
    solvent_buffer = solvent_config.get('buffer', 12.0) / 10 * unit.nanometers
    solventBoxVectors = computeBoxVectorsWithPadding(modeller.positions, solvent_buffer)
    modeller.positions = shiftToBoxCenter(modeller.positions, solventBoxVectors)
    modeller.topology.setPeriodicBoxVectors(solventBoxVectors)
    modeller.addSolvent(
        forcefield=ff,
        model=water_ff,
        neutralize=True,
        ionicStrength=solvent_config.get('ionic_strength', 0.0) * unit.molar
    )
    _save(modeller, 'solvent', **solvent_config)

    # Complex-phase
    if protein_pdb is not None:
        complex_buffer = complex_config.get('buffer', 15.0) / 10 * unit.nanometers
        pdb = app.PDBFile(str(protein_pdb))
        modeller = app.Modeller(app.Topology(), [])
        modeller.add(ligandA_struct.topology, ligandA_struct.positions)
        modeller.add(ligandB_struct.topology, ligandB_struct.positions)
        modeller.add(pdb.topology, pdb.positions)
        complexBoxVectors = computeBoxVectorsWithPadding(modeller.positions, complex_buffer)
        modeller.positions = shiftToBoxCenter(modeller.positions, complexBoxVectors)
        modeller.topology.setPeriodicBoxVectors(complexBoxVectors)
        modeller.addSolvent(
            forcefield=ff,
            model=water_ff,
            ionicStrength=solvent_config.get('ionic_strength', 0.15) * unit.molar,
            neutralize=True
        )
        _save(modeller, 'complex', **complex_config)