import os
import json
from pathlib import Path
from typing import Union, Dict, List, Tuple, Any

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.app.element as elem
import openmm.unit as unit
import parmed
from rdkit import Chem
from scipy.spatial.distance import cdist
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


def computeBoxVectorsWithPadding(positions: unit.Quantity, buffer: unit.Quantity, boxShape: str = 'cubic'):
    min_x, min_y, min_z = min(pos.x for pos in positions), min(pos.y for pos in positions), min(pos.z for pos in positions)
    max_x, max_y, max_z = max(pos.x for pos in positions), max(pos.y for pos in positions), max(pos.z for pos in positions)
    buffer = buffer.value_in_unit(unit.nanometer)
    box_x = (max_x - min_x) + buffer * 2
    box_y = (max_y - min_y) + buffer * 2
    box_z = (max_z - min_z) + buffer * 2
    if boxShape == 'cubic':
        box_x = box_y = box_z = max(box_x, box_y, box_z)
    return (
        mm.Vec3(box_x, 0.0, 0.0),
        mm.Vec3(0.0, box_y, 0.0),
        mm.Vec3(0.0, 0.0, box_z)
    )


def shiftToBoxCenter(positions: unit.Quantity, boxVectors: Tuple[mm.Vec3], returnShiftVec: bool = False):
    positions = positions.value_in_unit(unit.nanometers)
    boxCenter = (boxVectors[0] + boxVectors[1] + boxVectors[2]) / 2
    minVec = mm.Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
    maxVec = mm.Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
    posCenter = (minVec + maxVec) * 0.5
    shiftVec = boxCenter - posCenter
    newPositions = shiftPositions(positions, shiftVec)
    if returnShiftVec:
        return newPositions, shiftVec
    else:
        return newPositions


def shiftPositions(positions: unit.Quantity, shiftVec: np.ndarray):
    newPositions = unit.Quantity(value=[pos + shiftVec for pos in positions], unit=unit.nanometers)
    return newPositions


def generate_amber_mask(natomsA: int, natomsB: int, mapping: Dict[int, int], alchemical_water_info: Dict[str, List] = dict()):
    scA = [i for i in range(natomsA) if i not in mapping.keys()]
    scB = [i for i in range(natomsB) if i not in mapping.values()]
    res = {
        "noshakemask": f"@1-{natomsA+natomsB}",
        "timask1": f"@1-{natomsA}",
        "timask2": f"@{natomsA+1}-{natomsA+natomsB}",
        "scmask1": "@{}".format(','.join(str(i+1) for i in scA)),
        "scmask2": "@{}".format(','.join(str(i+1+natomsA) for i in scB))
    }
    if alchemical_water_info:
        alchemical_water_mask = {
            "noshakemask": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_water_oxygen'] + alchemical_water_info['alchemical_water_hydrogen'] + alchemical_water_info['alchemical_ions']),
            "timask1": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_water_oxygen'] + alchemical_water_info['alchemical_water_hydrogen']),
            "timask2": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_ions']),
            "scmask1": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_water_hydrogen'])
        }
        for key in alchemical_water_mask:
            if not res[key]:
                res[key] = alchemical_water_mask[key]
            else:
                res[key] = f'{res[key]},{alchemical_water_mask[key]}'
    # add single quote mark
    for key in res:
        if res[key].startswith('@,'):
            res[key] = '@' + res[key][2:]
        res[key] = f"'{res[key]}'"
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


def sanitize_water(struct: parmed.Structure):
    """
    Make the water topology correct for Amber simulation

    In Amber, TIP3P water will have 3 bonds, 0 angles. If not, pmemd will raise an error:
    `Error: Fast 3-point water residue, name and bond data incorrect!`. Also, water molecules has to be 
    named "WAT" to let Amber use SETTLE for waters. These are essential if one wants to use larger
    time step.
    """
    for residx, residue in enumerate(struct.residues):
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


def do_co_alchemical_water(modeller: app.Modeller, d_charge: int, scIndices: List[int], positiveIon: str = 'Na+', negativeIon: str = 'Cl-'):
    top, pos = modeller.topology, modeller.positions
    posNumpy = np.array([[p.x, p.y, p.z] for p in pos])
    posIonElements = {
        'Cs+': elem.cesium, 'K+': elem.potassium, 
        'Li+': elem.lithium, 'Na+': elem.sodium,
        'Rb+': elem.rubidium
    }
    negIonElements = {
        'Cl-': elem.chlorine, 'Br-': elem.bromine,
        'F-': elem.fluorine, 'I-': elem.iodine
    }

    if d_charge > 0:
        # counter ions
        element = negIonElements[negativeIon]
    else:
        element = posIonElements[positiveIon]

    atoms = list(top.atoms())
    
    scPositions = posNumpy[scIndices]

    waterIndices = np.array([atom.index for atom in atoms if atom.residue.name == 'HOH' and atom.name == 'O'])
    waterPositions = posNumpy[waterIndices]

    selectedIndices = []
    min_dist = np.min(cdist(waterPositions, scPositions), axis=1)
    waterIndicesWithDist = [(index, dist) for index, dist in zip(waterIndices, min_dist)]
    waterIndicesWithDist.sort(key=lambda x: x[1])
    for index, dist in waterIndicesWithDist:
        if dist < 2.0:
            continue
        if selectedIndices and np.linalg.norm(posNumpy[selectedIndices] - posNumpy[index], axis=1).min() > 0.5:
            selectedIndices.append(index)
        elif len(selectedIndices) == 0:
            selectedIndices.append(index)
        if len(selectedIndices) == abs(d_charge):
            break

    info = {
        "alchemical_ions": [],
        "alchemical_water_residues": [],
        "alchemical_water_oxygen": [],
        "alchemical_water_hydrogen": []
    }
    for index in selectedIndices:
        ionChain = top.addChain()
        ionResidue = top.addResidue(element.symbol.upper(), ionChain)
        ion = top.addAtom(element.symbol, element, ionResidue)
        pos.append(pos[index])
        info['alchemical_ions'].append(ion.index)
        water = atoms[index].residue
        info['alchemical_water_residues'].append(water.index)
        for atom in water.atoms():
            if atom.element.symbol == 'O':
                info['alchemical_water_oxygen'].append(atom.index)
            else:
                info['alchemical_water_hydrogen'].append(atom.index)
    
    modeller.topology = top
    modeller.positions = pos
    return info
    

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
    complex_config: Dict[str, Any] = dict(),
    use_charge_change: bool = True,
    use_settle_for_alchemical_water: bool = True
):
    # Create working directory
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)

    if not isinstance(ligandA_mol, Chem.Mol):
        ligandA_mol = Chem.SDMolSupplier(ligandA_mol, removeHs=False)[0]
    if not isinstance(ligandB_mol, Chem.Mol):
        ligandB_mol = Chem.SDMolSupplier(ligandB_mol, removeHs=False)[0]

    ligandA_charge = sum([at.GetFormalCharge() for at in ligandA_mol.GetAtoms()])
    ligandB_charge = sum([at.GetFormalCharge() for at in ligandB_mol.GetAtoms()])

    d_charge = ligandB_charge - ligandA_charge
    use_charge_change = use_charge_change and (d_charge != 0)
    
    ligandA_struct = parmed.load_file(str(ligandA_top))
    ligandA_struct.residues[0].name = 'MOLA'
    ligandA_struct.residues[0].resname = 'MOLA'
    ligandA_struct.coordinates = ligandA_mol.GetConformer().GetPositions()
    for atom in ligandA_struct.atoms:
        atom.name = f'{atom.name}a'

    ligandB_struct = parmed.load_file(str(ligandB_top))
    ligandB_struct.residues[0].name = 'MOLB'
    ligandB_struct.coordinates = ligandB_mol.GetConformer().GetPositions()
    for atom in ligandB_struct.atoms:
        atom.name = f'{atom.name}b'

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
    num_atoms_A, num_atoms_B, mapping_renum = ligandA_mol.GetNumAtoms(), ligandB_mol.GetNumAtoms(), {i:i for i in range(len(mapping))}

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
        # This is for cases if A and B are stereo-isomers
        residueTemplates = {}
        for res in modeller.topology.residues():
            if res.name in ['MOLA', 'MOLB']:
                residueTemplates[res] = res.name
            if all(name in residueTemplates for name in ['MOLA', 'MOLB']):
                break

        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False, residueTemplates=residueTemplates)
        tmp = parmed.openmm.load_topology(modeller.topology, system, xyz=modeller.positions)

        # Note: In AmberTools, TIP3P water will have 3 bonds, 0 angles. If not, pmemd will raise an error:
        # `Error: Fast 3-point water residue, name and bond data incorrect!` 
        for residx, residue in enumerate(tmp.residues):
            # Give alchemical water a unique name
            if (not use_settle_for_alchemical_water) and residx in kwargs.get('alchemical_water_residues', []):
                residue.name = 'ALW'
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

        # Hydrogen mass repartition
        if do_hmr:
            hydrogen_mass_repartition(tmp, hydrogen_mass, do_hmr_water)

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
    mask = generate_amber_mask(num_atoms_A, num_atoms_B, mapping_renum)
    gas_config.update(mask)
    _save(modeller, 'gas', **gas_config)

    scIndices = mask['scmask1'].strip("'")[1:].split(',') + mask['scmask2'].strip("'")[1:].split(',')
    scIndices = [int(x) - 1 for x in scIndices if x]
    
    # Solvent phase
    solvent_buffer = solvent_config.get('buffer', 12.0) / 10 * unit.nanometers
    solventBoxVectors = computeBoxVectorsWithPadding(modeller.positions, solvent_buffer)
    modeller.positions = shiftToBoxCenter(modeller.positions, solventBoxVectors)
    modeller.topology.setPeriodicBoxVectors(solventBoxVectors)
    modeller.addSolvent(
        forcefield=ff,
        model=water_ff,
        neutralize=True,
        ionicStrength=solvent_config.get('ionic_strength', 0.0) * unit.molar,
        residueTemplates={res: res.name for res in modeller.topology.residues() if res.name in ['MOLA', 'MOLB']}
    )

    if use_charge_change:
        alchem_water_info = do_co_alchemical_water(modeller, d_charge, scIndices)
        mdin_mod = {
            'solvent': ['&wt TYPE="DUMPFREQ", istep1=100, /', '&wt TYPE="END", /', 'DUMPAVE=dist'], 
            'complex': ['&wt TYPE="DUMPFREQ", istep1=100, /', '&wt TYPE="END", /', 'DUMPAVE=dist']
        }
    else:
        alchem_water_info = dict()
        mdin_mod = dict()

    _save(modeller, 'solvent', **solvent_config, **alchem_water_info)
    mask = generate_amber_mask(num_atoms_A, num_atoms_B, mapping_renum, alchem_water_info)
    if use_charge_change:
        waterOIndex = mask['timask1'].strip("'").strip('@').split(',')[-3]
        ionIndex = mask['timask2'].strip("'").strip('@').split(',')[-1]
        scmask1 = mask['scmask1'].strip("'").strip('@').split(',')[:-2]
        scmask2 = mask['scmask2'].strip("'").strip('@').split(',')

        # determine the anchor on the sc region
        # "anchor" defines the distance restraint between the ligand and the alchemical water
        positions = np.array(modeller.positions.value_in_unit(unit.nanometers))
        if len(scmask1) == 0:
            anchor2 = scmask2[0]
            anchor1 = np.argmin(np.linalg.norm(
                positions[:num_atoms_A] - positions[int(anchor2) - 1],
                axis=1
            )) + 1
        elif len(scmask2) == 0:
            anchor1 = scmask1[0]
            anchor2 = np.argmin(np.linalg.norm(
                positions[num_atoms_A:num_atoms_A+num_atoms_B] - positions[int(anchor1) - 1],
                axis=1
            )) + num_atoms_A
        else:
            anchor1 = scmask1[0]
            anchor2 = scmask2[0]

        maxLen = min([
            np.linalg.norm([solventBoxVectors[0].x, solventBoxVectors[0].y, solventBoxVectors[0].z]),
            np.linalg.norm([solventBoxVectors[1].x, solventBoxVectors[1].y, solventBoxVectors[1].z]),
            np.linalg.norm([solventBoxVectors[2].x, solventBoxVectors[2].y, solventBoxVectors[2].z])
        ]) * 10
        r1, r2 = 10.0, 15.0
        assert maxLen > 2 * r2, "The box is too small for charge change FEP"
        mdin_mod['solvent'] += [
            f'&rst iat={anchor1},{waterOIndex}, r1={r1:.2f}, r2={r2:.2f}, r3={maxLen - r2:.2f}, r4={maxLen - r1:.2f}, rk2=1000.0, rk3=1000.0, /',
            f'&rst iat={anchor2},{ionIndex}, r1={r1:.2f}, r2={r2:.2f}, r3={maxLen - r2:.2f}, r4={maxLen - r1:.2f}, rk2=1000.0, rk3=1000.0, /'
        ]
    
    solvent_config.update(mask)

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
            neutralize=True,
            residueTemplates={res: res.name for res in modeller.topology.residues() if res.name in ['MOLA', 'MOLB']}
        )
        if use_charge_change:
            alchem_water_info = do_co_alchemical_water(modeller, d_charge, scIndices)
        else:
            alchem_water_info = dict()

        _save(modeller, 'complex', **complex_config, **alchem_water_info)
        mask = generate_amber_mask(num_atoms_A, num_atoms_B, mapping_renum, alchem_water_info)
        if use_charge_change:
            waterOIndex = mask['timask1'].strip("'").strip('@').split(',')[-3]
            ionIndex = mask['timask2'].strip("'").strip('@').split(',')[-1]
            scmask1 = mask['scmask1'].strip("'").strip('@').split(',')[:-2]
            scmask2 = mask['scmask2'].strip("'").strip('@').split(',')
            
            # determine the anchor on the sc region
            # "anchor" defines the distance restraint between the ligand and the alchemical water
            positions = np.array(modeller.positions.value_in_unit(unit.nanometers))
            if len(scmask1) == 0:
                anchor2 = scmask2[0]
                anchor1 = np.argmin(np.linalg.norm(
                    positions[:num_atoms_A] - positions[int(anchor2) - 1],
                    axis=1
                )) + 1
            elif len(scmask2) == 0:
                anchor1 = scmask1[0]
                anchor2 = np.argmin(np.linalg.norm(
                    positions[num_atoms_A:num_atoms_A+num_atoms_B] - positions[int(anchor1) - 1],
                    axis=1
                )) + num_atoms_A
            else:
                anchor1 = scmask1[0]
                anchor2 = scmask2[0]

            maxLen = min([
                np.linalg.norm([complexBoxVectors[0].x, complexBoxVectors[0].y, complexBoxVectors[0].z]),
                np.linalg.norm([complexBoxVectors[1].x, complexBoxVectors[1].y, complexBoxVectors[1].z]),
                np.linalg.norm([complexBoxVectors[2].x, complexBoxVectors[2].y, complexBoxVectors[2].z])
            ]) * 10
            r1, r2 = 10.0, 15.0
            assert maxLen > 2 * r2, "The box is too small for charge change FEP"
            mdin_mod['complex'] += [
                f'&rst iat={anchor1},{waterOIndex}, r1={r1:.2f}, r2={r2:.2f}, r3={maxLen - r2:.2f}, r4={maxLen - r1:.2f}, rk2=1000.0, rk3=1000.0, /',
                f'&rst iat={anchor2},{ionIndex}, r1={r1:.2f}, r2={r2:.2f}, r3={maxLen - r2:.2f}, r4={maxLen - r1:.2f}, rk2=1000.0, rk3=1000.0, /'
            ]
        complex_config.update(mask)   
    return mdin_mod