from __future__ import annotations

__all__ = [
    'FF_XMLS', 'do_co_alchemical_water', 'sanitize_water', 'compute_net_charge_from_openmm_system',
    'hydrogen_mass_repartition', 'computeBoxVectorsWithPadding', 'shiftToBoxCenter', 'shiftPositions',
    'fix_excess_charge', 'set_alchemical_water_restraints', 'generate_amber_mask'
]

import math
from typing import Union, Dict, List, Tuple, Any
import logging
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.app.element as elem
import openmm.unit as unit
import parmed
from scipy.spatial.distance import cdist

from ..config import AmberRstSettings

logger = logging.getLogger(__name__)


FF_XMLS = {
    'ff14SB': 'amber14/protein.ff14SB.xml',
    'tip3p': 'amber14/tip3p.xml'
}


def _find_ion(top, ion='NA'):
    _find_residue = None
    for residue in reversed(list(top.residues())):
        if residue.name == ion:
            _find_residue = residue
            break
    else:
        raise RuntimeError(f"Not find ion {ion}")
    return [_find_residue]    


def fix_excess_charge(modeller: app.Modeller, stateB_charge: int):
    '''OpenMM modeller neutralize the whole system, which means the charges from both state A and B 
    are taken in to account. But, we only want state A or B to be neutralized. That means if state B has a net charge
    the state A or B will have an access charge'''
    if stateB_charge > 0:
        modeller.delete(_find_ion(modeller.topology, 'CL'))
    elif stateB_charge < 0:
        modeller.delete(_find_ion(modeller.topology, 'NA'))


def do_co_alchemical_water(modeller: app.Modeller, d_charge: int, scIndices: List[int], proteinIndices: List[int] = list(), positiveIon: str = 'Na+', negativeIon: str = 'Cl-', dist_thresh: float = 15.0):
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

    element = negIonElements[negativeIon] if d_charge > 0 else posIonElements[positiveIon]

    atoms = list(top.atoms())
    
    scPositions = posNumpy[scIndices]

    waterIndices = np.array([atom.index for atom in atoms if atom.residue.name == 'HOH' and atom.name == 'O'])
    waterPositions = posNumpy[waterIndices]

    selectedIndices = []
    min_dist = np.min(cdist(waterPositions, scPositions), axis=1)

    if len(proteinIndices) > 0:
        min_dist_to_protein = np.min(cdist(waterPositions, posNumpy[proteinIndices]), axis=1)
    else:
        min_dist_to_protein = np.full(len(waterIndices), np.inf)

    waterIndicesWithDist = [(index, dist, dist_to_protein) for index, dist, dist_to_protein in zip(waterIndices, min_dist, min_dist_to_protein)]
    waterIndicesWithDist.sort(key=lambda x: x[1], reverse=True)

    # half box in nm
    boxVectors = modeller.topology.getPeriodicBoxVectors()
    halfBox = min([
        np.linalg.norm([boxVectors[0].x, boxVectors[0].y, boxVectors[0].z]),
        np.linalg.norm([boxVectors[1].x, boxVectors[1].y, boxVectors[1].z]),
        np.linalg.norm([boxVectors[2].x, boxVectors[2].y, boxVectors[2].z])
    ]) / 2 

    for index, dist, dist_to_protein in waterIndicesWithDist:
        if dist > halfBox:
            continue
        if dist < (dist_thresh / 10) or dist_to_protein < (dist_thresh / 10):
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


def set_alchemical_water_restraints(modeller: app.Modeller, scIndices: List[int], alchem_water_info: Dict[str, List[int]], dist_thresh: float = 15.0, k: float = 1000.0):
    """
    Keep alchemical water away from soft core atoms, otherwise thy will collide

    scIndices: soft-core atoms
    """
    boxVectors = modeller.topology.getPeriodicBoxVectors()
    halfBox = min([
        np.linalg.norm([boxVectors[0].x, boxVectors[0].y, boxVectors[0].z]),
        np.linalg.norm([boxVectors[1].x, boxVectors[1].y, boxVectors[1].z]),
        np.linalg.norm([boxVectors[2].x, boxVectors[2].y, boxVectors[2].z])
    ]) / 2 * 10
    rst_settings = []
    atoms = list(modeller.topology.atoms())
    logger.info(f'Distance restraints will be applied to restrain the co-alchemical water oxygen within ({dist_thresh:.2f}~{halfBox:.2f}) Angstrom from the soft-core atoms')
    assert halfBox >= 5.0 + dist_thresh, f"The box is too small for charge change FEP ({halfBox*2:.2f}) Ang but at least {10.0+2*dist_thresh:.2f} recommended."
    for oxy_idx, ion_idx in zip(alchem_water_info['alchemical_water_oxygen'], alchem_water_info['alchemical_ions']):
        logger.info(f"CO-ALCHEMICAL WATER created: oxygen #{oxy_idx} -> {atoms[ion_idx].name}")

        for sc in scIndices:
            rst_settings.append(AmberRstSettings(iat=[sc+1, oxy_idx+1], r1=dist_thresh-5.0, r2=dist_thresh, r3=halfBox, r4=halfBox+5.0, rk2=k, rk3=k))
            rst_settings.append(AmberRstSettings(iat=[sc+1, ion_idx+1], r1=dist_thresh-5.0, r2=dist_thresh, r3=halfBox, r4=halfBox+5.0, rk2=k, rk3=k))
            sc_pos = np.array([modeller.positions[sc][0], modeller.positions[sc][1], modeller.positions[sc][2]])
            wat_pos = np.array([modeller.positions[oxy_idx][0], modeller.positions[oxy_idx][1], modeller.positions[oxy_idx][2]])
            logger.info(f'Co-alchemical water oxygen ({oxy_idx}) to soft-core atom ({sc}) distance: {np.linalg.norm(sc_pos - wat_pos)._value*10:.2f} Angstrom')
    
    return rst_settings


def sanitize_water(struct: parmed.Structure, specical_water_resname: str = 'ALW', special_water_residx: List[int] = list()):
    '''In AmberTools, TIP3P water will have 3 bonds, 0 angles. If not, pmemd will raise an error:
    ``Error: Fast 3-point water residue, name and bond data incorrect!``
    Therefore, we need to correct them after OpenMM preparation
    '''
    for residx, residue in enumerate(struct.residues):
        if residx in special_water_residx:
            residue.name = specical_water_resname
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


def compute_net_charge_from_openmm_system(system: mm.System):
    """
    Compute the net charge of an OpenMM System.
    
    Parameters
    ----------
    system : openmm.System
        The OpenMM System to compute the net charge for.
    
    Returns
    -------
    float
        The net charge of the system.
    """
    net_charge = 0.0
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            for i in range(force.getNumParticles()):
                charge = force.getParticleParameters(i)[0]
                net_charge += charge.value_in_unit(unit.elementary_charge)
            break
    return round(net_charge)


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


def computeBoxVectorsWithPadding(positions: unit.Quantity, buffer: unit.Quantity, boxShape: str = 'cube'):
    positions = positions.value_in_unit(unit.nanometer)
    buffer = buffer.value_in_unit(unit.nanometer)
    minVec = mm.Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
    maxVec = mm.Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
    center = (minVec + maxVec) * 0.5
    radius = max(unit.norm(center-pos) for pos in positions)
    width = max(2*buffer, 2*radius+buffer)

    if boxShape == 'cube':
        return (mm.Vec3(width, 0, 0), mm.Vec3(0, width, 0), mm.Vec3(0, 0, width))
    elif boxShape == 'dodecahedron':
        return (mm.Vec3(width, 0, 0), mm.Vec3(0, width, 0), mm.Vec3(0.5, 0.5, 0.5*math.sqrt(2))*width)
    elif boxShape == 'octahedron':
        return (mm.Vec3(width, 0, 0), mm.Vec3(1/3, 2*math.sqrt(2)/3, 0)*width, mm.Vec3(-1/3, math.sqrt(2)/3, math.sqrt(6)/3)*width)
    else:
        raise ValueError(f'Illegal box shape: {boxShape}')


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


def shiftPositions(positions: unit.Quantity | List, shiftVec: np.ndarray):
    if isinstance(positions, unit.Quantity):
        positions = positions.value_in_unit(unit.nanometers)
    newPositions = unit.Quantity(value=[pos + shiftVec for pos in positions], unit=unit.nanometers)
    return newPositions


def generate_amber_mask(natomsA: int, natomsB: int, mapping: dict[int, int], alchemical_water_info: dict[str, list] = dict(), mode: str = 'rbfe'):
    if mode == 'rbfe':
        scA = [i for i in range(natomsA) if i not in mapping.keys()]
        scB = [i for i in range(natomsB) if i not in mapping.values()]
        res = {
            # "noshakemask": f"@1-{natomsA+natomsB}",
            "timask1": f"@1-{natomsA}",
            "timask2": f"@{natomsA+1}-{natomsA+natomsB}",
            "scmask1": "@{}".format(','.join(str(i+1) for i in scA)),
            "scmask2": "@{}".format(','.join(str(i+1+natomsA) for i in scB))
        }
    elif mode == 'abfe':
        res = {
            # "noshakemask": f"@1-{natomsA+natomsB}",
            "timask1": f"@1-{natomsA}",
            "timask2": "",
            "scmask1": f"@1-{natomsA}",
            "scmask2": ""
        }
    elif mode == 'abfe_restr':
        res = {
            "timask1": f"@1-{natomsA}",
            "timask2": f"@{natomsA+1}-{2*natomsA}",
            "scmask1": "",
            "scmask2": ""
        }
    else:
        raise ValueError(f"unrecognized mode: {mode}")
    if alchemical_water_info:
        alchemical_water_mask = {
            # "noshakemask": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_water_oxygen'] + alchemical_water_info['alchemical_water_hydrogen'] + alchemical_water_info['alchemical_ions']),
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