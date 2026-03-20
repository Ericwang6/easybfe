from __future__ import annotations

__all__ = [
    'FF_XMLS', 'create_alchemical_ions', 'sanitize_water', 'compute_net_charge_from_openmm_system',
    'hydrogen_mass_repartition', 'computeBoxVectorsWithPadding', 'shiftToBoxCenter', 'shiftPositions',
    'fix_excess_charge', 'set_alchemical_water_restraints', 'generate_amber_mask'
]

import math
from typing import Union, Dict, List, Tuple, Optional, Any, Literal
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


def create_alchemical_ions(
    modeller: app.Modeller,
    stateA_charge: int,
    stateB_charge: int,
    scIndices: list[int],
    soluteIndices: Optional[list[int]] = None,
    positiveIon: str = "Na+",
    negativeIon: str = "Cl-",
    method: Literal["dummy_ion", "coalchem_water"] = "coalchem_water",
):
    top = modeller.topology
    pos = modeller.positions
    # Work in Angstroms for geometry, convert back to nanometers at the end
    posNumpy = np.array([[p.x, p.y, p.z] for p in pos]) * 10

    if stateA_charge == 0 and stateB_charge == -1:
        ion, mode = positiveIon, '2i'
    elif stateA_charge == -1 and stateB_charge == 0:
        ion, mode = positiveIon, 'i2'
    elif stateA_charge == 0 and stateB_charge == 1:
        ion, mode = negativeIon, '2i'
    elif stateA_charge == 1 and stateB_charge == 0:
        ion, mode = negativeIon, 'i2'
    else:
        raise RuntimeError(f"Charge change from {stateA_charge} to {stateB_charge} too drastic, which is not allowed.")

    ionResidueName = ion.strip('+-').upper()
    solventResidue = (positiveIon.strip('+-').upper(), negativeIon.strip('+-').upper(), 'WAT', 'HOH')

    ionIndices = np.array([at.index for at in top.atoms() if at.residue.name == ionResidueName])
    ionPositions = posNumpy[ionIndices]
    soluteIndices = soluteIndices if soluteIndices else [at.index for at in top.atoms() if at.residue.name not in solventResidue]
    soluteIndices = posNumpy[soluteIndices]
    scPositions = posNumpy[scIndices]

    # find an ion that is far-away enough from both sc region and solute
    # half box in angstrom
    boxVectors = modeller.topology.getPeriodicBoxVectors()
    halfBox = min([
        np.linalg.norm([boxVectors[0].x, boxVectors[0].y, boxVectors[0].z]),
        np.linalg.norm([boxVectors[1].x, boxVectors[1].y, boxVectors[1].z]),
        np.linalg.norm([boxVectors[2].x, boxVectors[2].y, boxVectors[2].z])
    ]) / 2 * 10

    distIonSc = np.min(cdist(ionPositions, scPositions), axis=1)
    distIonSolute = np.min(cdist(ionPositions, soluteIndices), axis=1)
    
    argmax = np.argmax(distIonSolute)
    ionIndex = ionIndices[argmax]
    distsc = np.linalg.norm(posNumpy[scIndices[0]] - posNumpy[ionIndex])
    logger.info(f"Atom #{ionIndex} ({ion}, 0-indexed) is alchemically changed. Min distance to sc region atom {scIndices[0]}: {distsc:.2f} Angstrom")

    info = {
        "alchemical_ions": [ionIndex],
        "alchemical_water_residues": [],
        "alchemical_water_oxygen": [],
        "alchemical_water_hydrogen": [],
        "mode": mode,
        "timask1": '',
        "timask2": '',
        "scmask1": '',
        "scmask2": '',
        "dist": distsc,
        'method': method
    }

    ionMask = f"@{ionIndex+1}"
    if method == "coalchem_water":
        # find a water position from the modeller
        waterAtoms = []
        waterBonds = []
        for res in top.residues():
            if res.name == 'HOH':
                waterAtoms = [at for at in res.atoms()]
                waterBonds = [(bo.atom1.index, bo.atom2.index) for bo in res.bonds()]
                break
        
        o = [at.index for at in waterAtoms if at.name.startswith('O')][0]
        waterBonds = [(a1-o, a2-o) for a1, a2 in waterBonds]
        
        # translational move to the ion
        waterPos = posNumpy[[at.index for at in waterAtoms]]
        shift = waterPos[0] - posNumpy[ionIndex]
        waterPos -= shift

        waterChain = top.addChain()
        waterResidue = top.addResidue("HOH", waterChain)
        info["alchemical_water_residues"].append(waterResidue.index)
        waterAtomsNew = []
        for at, apos in zip(waterAtoms, waterPos):
            atom = top.addAtom(at.name, at.element, waterResidue)
            waterAtomsNew.append(atom)
            if at.name.startswith("O"):
                info["alchemical_water_oxygen"].append(atom.index)
            else:
                info["alchemical_water_hydrogen"].append(atom.index)

        waterMask = "@" + ",".join([str(at.index + 1) for at in waterAtomsNew])
        scMask = "@" + ",".join(str(i + 1) for i in info["alchemical_water_hydrogen"])

        for a1, a2 in waterBonds:
            top.addBond(waterAtomsNew[a1], waterAtomsNew[a2])
        
        modeller.topology = top
        # Append new water coordinates in Angstroms, then convert back to nanometers
        modeller.positions = unit.Quantity(
            np.concatenate([posNumpy, waterPos], axis=0)/10, unit=unit.nanometer
        )   

        if mode == "i2":
            info["timask1"], info["timask2"], info["scmask2"] = ionMask, waterMask, scMask
        else:
            info["timask1"], info["timask2"], info["scmask1"] = waterMask, ionMask, scMask
    else:
        if mode == "i2":
            info["timask1"], info["timask2"], info["scmask1"] = ionMask, "", ionMask
        else:
            info["timask1"], info["timask2"], info["scmask2"] = "", ionMask, ionMask

    return info


# def set_alchemical_water_restraints(modeller: app.Modeller, scIndices: List[int], alchem_water_info: Dict[str, List[int]], dist_thresh: float = 15.0, k: float = 1000.0):
#     """
#     Keep alchemical water away from soft core atoms, otherwise thy will collide

#     scIndices: soft-core atoms
#     """
#     boxVectors = modeller.topology.getPeriodicBoxVectors()
#     halfBox = min([
#         np.linalg.norm([boxVectors[0].x, boxVectors[0].y, boxVectors[0].z]),
#         np.linalg.norm([boxVectors[1].x, boxVectors[1].y, boxVectors[1].z]),
#         np.linalg.norm([boxVectors[2].x, boxVectors[2].y, boxVectors[2].z])
#     ]) / 2 * 10
#     rst_settings = []
#     atoms = list(modeller.topology.atoms())
#     logger.info(f'Distance restraints will be applied to restrain the co-alchemical water oxygen within ({dist_thresh:.2f}~{halfBox:.2f}) Angstrom from the soft-core atoms')
#     assert halfBox >= 5.0 + dist_thresh, f"The box is too small for charge change FEP ({halfBox*2:.2f}) Ang but at least {10.0+2*dist_thresh:.2f} recommended."
#     for oxy_idx, ion_idx in zip(alchem_water_info['alchemical_water_oxygen'], alchem_water_info['alchemical_ions']):
#         logger.info(f"CO-ALCHEMICAL WATER created: oxygen #{oxy_idx} -> {atoms[ion_idx].name}")

#         for sc in scIndices:
#             rst_settings.append(AmberRstSettings(iat=[sc+1, oxy_idx+1], r1=dist_thresh-5.0, r2=dist_thresh, r3=halfBox, r4=halfBox+5.0, rk2=k, rk3=k))
#             rst_settings.append(AmberRstSettings(iat=[sc+1, ion_idx+1], r1=dist_thresh-5.0, r2=dist_thresh, r3=halfBox, r4=halfBox+5.0, rk2=k, rk3=k))
#             sc_pos = np.array([modeller.positions[sc][0], modeller.positions[sc][1], modeller.positions[sc][2]])
#             wat_pos = np.array([modeller.positions[oxy_idx][0], modeller.positions[oxy_idx][1], modeller.positions[oxy_idx][2]])
#             logger.info(f'Co-alchemical water oxygen ({oxy_idx}) to soft-core atom ({sc}) distance: {np.linalg.norm(sc_pos - wat_pos)._value*10:.2f} Angstrom')
    
#     return rst_settings


def set_alchemical_water_restraints(modeller: app.Modeller, scIndices: list[int], coion_info: dict[str, Any], buffer = 10.0, k: float = 1000.0):
    """
    Keep alchemical water away from soft core atoms, otherwise thy will collide

    scIndices: soft-core atoms
    """
    # boxVectors = modeller.topology.getPeriodicBoxVectors()
    # halfBox = min([
    #     np.linalg.norm([boxVectors[0].x, boxVectors[0].y, boxVectors[0].z]),
    #     np.linalg.norm([boxVectors[1].x, boxVectors[1].y, boxVectors[1].z]),
    #     np.linalg.norm([boxVectors[2].x, boxVectors[2].y, boxVectors[2].z])
    # ]) / 2 * 10
    rst_settings = []
    dist = coion_info['dist']
    assert dist >= buffer, f'Co-alchemical ion too close to sc region: {dist:.2f} < {buffer:.2f}' 
    # atoms = list(modeller.topology.atoms())
    # logger.info(f'Distance restraints will be applied to restrain the co-alchemical water oxygen within ({dist_thresh:.2f}~{halfBox:.2f}) Angstrom from the soft-core atoms')
    # assert halfBox >= 5.0 + dist_thresh, f"The box is too small for charge change FEP ({halfBox*2:.2f}) Ang but at least {10.0+2*dist_thresh:.2f} recommended."
    for oxy_idx, ion_idx in zip(coion_info['alchemical_water_oxygen'], coion_info['alchemical_ions']):
        sc = scIndices[0]
        rst_settings.append(AmberRstSettings(iat=[sc+1, oxy_idx+1], r1=dist-buffer, r2=dist-buffer/2, r3=dist+buffer/2, r4=dist+buffer, rk2=k, rk3=k))
        rst_settings.append(AmberRstSettings(iat=[sc+1, ion_idx+1], r1=dist-buffer, r2=dist-buffer/2, r3=dist+buffer/2, r4=dist+buffer, rk2=k, rk3=k))
    
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


def generate_amber_mask(natomsA: int, natomsB: int, mapping: dict[int, int], coion_info: dict[str, list] = dict(), mode: str = 'rbfe'):
    if mode == 'rbfe':
        scA = [i for i in range(natomsA) if i not in mapping.keys()]
        scB = [i for i in range(natomsB) if i not in mapping.values()]
        res = {
            # "noshakemask": f"@1-{natomsA+natomsB}",
            "timask1": f"@1-{natomsA}",
            "timask2": f"@{natomsA+1}-{natomsA+natomsB}",
            "scmask1": "@{}".format(','.join(str(i+1) for i in scA)) if len(scA) > 0 else '',
            "scmask2": "@{}".format(','.join(str(i+1+natomsA) for i in scB)) if len(scB) > 0 else ''
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
    update_mask(res, coion_info)
    return res


def update_mask(mask: dict[str, str], update: dict[str, Any]):
    for key in mask:
        ori = mask[key]
        upd = update.get(key, '')
        if not ori:
            mask[key] = upd
        elif upd:
            assert upd[0] == ori[0], f'Mask specification not same: {upd} startswith {upd[0]}, but {ori} startswith {ori[0]}'
            assert ori[0] in (':', '@'), f'Mask should startswith ":" or "@"'
            new = ori + ',' + update[key][1:]
            new = new.replace('@,', '@')
            if new == '@,' or new == '@':
                new = ''
            mask[key] = new
            