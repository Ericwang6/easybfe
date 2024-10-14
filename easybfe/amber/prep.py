import os
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom
from pathlib import Path
from typing import Union, Dict, Optional, List, Tuple
import logging

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from rdkit import Chem

import parmed
import parmed.unit as u
from parmed.periodic_table import Element as ELEMENT


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


class OpenmmXML:
    def __init__(self):
        self.ffxml = None
        self.topxml = None
    
    @staticmethod
    def to_pretty_xmlstr(xmlele: ET.Element):
        uglystr = ET.tostring(xmlele, "unicode")
        pretxml = xml.dom.minidom.parseString(uglystr)
        pretstr = pretxml.toprettyxml()
        return pretstr
    
    @staticmethod
    def to_pretty_xmlfile(xmlele: ET.Element, fname: os.PathLike):
        xmlstr = OpenmmXML.to_pretty_xmlstr(xmlele)
        with open(fname, 'w') as f:
            f.write(xmlstr)
    
    def write_ffxml(self, fname: os.PathLike):
        self.to_pretty_xmlfile(self.ffxml, fname)
    
    def write_topxml(self, fname: os.PathLike):
        self.to_pretty_xmlfile(self.topxml, fname)

    @classmethod
    def from_parmed(cls, struct: parmed.Structure, useAtrributeFromResidue: bool = True, onlyCharges: bool = False):
        if onlyCharges:
            assert useAtrributeFromResidue, "useAtrributeFromResidue must be True when onlyCharges is True"
        struct.join_dihedrals()
        ffxml = ET.Element("ForceField")
        topxml = ET.Element("Residues")
        
        # regularize atom names, and keep name unique
        atomNamesCount = {}
        atomNames = []
        atomTypes = []
        for atom in struct.atoms:
            if atom.name not in atomNamesCount:
                name = atom.name
            else:
                name = atom.name + "_" + str(atomNamesCount[atom.name] + 1)
                # This is bad for protein systems, but we will keep it as this function is for ligands
                raise NotImplementedError(f"Identical names: {atom.name}")
            atomNamesCount[atom.name] = atomNamesCount.get(atom.name, 0) + 1
            atomNames.append(name)
            atomTypes.append(atom.residue.name + '-' + name)

        # topology
        residuesElement = ET.SubElement(ffxml, "Residues")
        for res in struct.residues:
            resElement = ET.SubElement(residuesElement, "Residue")
            resElement.set("name", res.name)
            topresElement = ET.SubElement(topxml, "Residue")
            topresElement.set("name", res.name)
            for atom in res.atoms:
                atomElement = ET.SubElement(resElement, "Atom")
                atomElement.set("name", atomNames[atom.idx])
                atomElement.set("type", atomTypes[atom.idx])
                if useAtrributeFromResidue:
                    atomElement.set("charge", f"{atom.charge:.10f}")
            for bond in struct.bonds:
                bondElement = ET.SubElement(resElement, "Bond")
                bondElement.set("atomName1", atomNames[bond.atom1.idx])
                bondElement.set("atomName2", atomNames[bond.atom2.idx])
                topbondElement = ET.SubElement(topresElement, "Bond")
                topbondElement.set("from", atomNames[bond.atom1.idx])
                topbondElement.set("to", atomNames[bond.atom2.idx])
                
        # atoms
        atomTypesElement = ET.SubElement(ffxml, "AtomTypes")
        for atom in struct.atoms:
            atomElement = ET.SubElement(atomTypesElement, "Type")
            atomElement.set("element", ELEMENT[atom.atomic_number])
            atomElement.set("name", atomTypes[atom.idx])
            atomElement.set("class", atom.type)
            atomElement.set("mass", f"{atom.mass:.3f}")
        
        if not onlyCharges:
            # bonds
            bondsElement = ET.SubElement(ffxml, "HarmonicBondForce")
            conv = (u.kilocalorie_per_mole / u.angstrom ** 2).conversion_factor_to(
                u.kilojoule_per_mole / u.nanometer ** 2) * 2
            for bond in struct.bonds:
                bondElement = ET.SubElement(bondsElement, "Bond")
                length = bond.type.req / 10 # in nm
                k = bond.type.k * conv
                bondElement.set("type1", str(atomTypes[bond.atom1.idx]))
                bondElement.set("type2", str(atomTypes[bond.atom2.idx]))
                bondElement.set("length", f"{length:.10f}")
                bondElement.set("k", f"{k:.10f}")
            
            # angles
            anglesElement = ET.SubElement(ffxml, "HarmonicAngleForce")
            conv = (u.kilocalorie_per_mole/u.radian ** 2).conversion_factor_to(
                    u.kilojoule_per_mole/u.radian ** 2 ) * 2
            deg2rad = u.degree.conversion_factor_to(u.radian)
            for angle in struct.angles:
                angleElement = ET.SubElement(anglesElement, "Angle")
                thetaeq = angle.type.theteq * deg2rad # in rad
                k = angle.type.k * conv
                angleElement.set("type1", str(atomTypes[angle.atom1.idx]))
                angleElement.set("type2", str(atomTypes[angle.atom2.idx]))
                angleElement.set("type3", str(atomTypes[angle.atom3.idx]))
                angleElement.set("angle", f"{thetaeq:.10f}")
                angleElement.set("k", f"{k:.10f}")
            
            # dihedrals
            dihesElement = ET.SubElement(ffxml, "PeriodicTorsionForce")
            dihesElement.set("ordering", "amber")
            conv = u.kilocalories.conversion_factor_to(u.kilojoules)
            for dihe in struct.dihedrals:
                if dihe.improper:
                    # OpenMM improper definitions is a little bit different
                    diheElement = ET.SubElement(dihesElement, "Improper")
                    diheElement.set("type1", str(atomTypes[dihe.atom3.idx]))
                    diheElement.set("type2", str(atomTypes[dihe.atom1.idx]))
                    diheElement.set("type3", str(atomTypes[dihe.atom2.idx]))
                    diheElement.set("type4", str(atomTypes[dihe.atom4.idx]))
                else:
                    diheElement = ET.SubElement(dihesElement, "Proper")
                    diheElement.set("type1", str(atomTypes[dihe.atom1.idx]))
                    diheElement.set("type2", str(atomTypes[dihe.atom2.idx]))
                    diheElement.set("type3", str(atomTypes[dihe.atom3.idx]))
                    diheElement.set("type4", str(atomTypes[dihe.atom4.idx]))

                def _set_dihedral(dtype, num):
                    diheElement.set(f"periodicity{num}", str(dtype.per))
                    diheElement.set(f"phase{num}", f"{dtype.phase * deg2rad:.10f}")
                    diheElement.set(f"k{num}", f"{dtype.phi_k * conv:.10f}")

                if isinstance(dihe.type, parmed.DihedralType):
                    _set_dihedral(dihe.type, 1)
                elif isinstance(dihe.type, parmed.DihedralTypeList):
                    for i, dtype in enumerate(dihe.type):
                        _set_dihedral(dtype, i+1)
                else:
                    raise TypeError("Unknown dihedral type")
            
            # nonbonded terms
            nbsElement = ET.SubElement(ffxml, "NonbondedForce")
            if hasattr(struct, "defaults"):
                assert struct.defaults.gen_pairs == 'yes'
                # In GMX top, 1-4 scaling factors are stored in [ defaults ] section
                if abs(struct.defaults.fudgeQQ - 5 / 6) <= 1e-3:
                    coul14scale = 5 / 6 # amber
                elif abs(struct.defaults.fudgeQQ - 1 / 2) <= 1e-3:
                    coul14scale = 1 / 2 # opls
                else:
                    coul14scale = struct.defaults.fudgeQQ
                if abs(struct.defaults.fudgeLJ - 1 / 2) <= 1e-3:
                    lj14scale = 1 / 2
                else:
                    lj14scale = struct.defaults.fudgeLJ
            else:
                # In Amber frcmod, 1-4 scaling factores are stored in each dihedral type
                scee = np.array([dtype.scee for dtypelist in struct.dihedral_types for dtype in dtypelist if dtype.scee != 0.0])
                scnb = np.array([dtype.scnb for dtypelist in struct.dihedral_types for dtype in dtypelist if dtype.scnb != 0.0])
                if np.allclose(scee, 1.2):
                    coul14scale = 5 / 6
                else:
                    raise RuntimeError("Undetermined coul 1-4 scale")
                
                if np.allclose(scnb, 2.0):
                    lj14scale = 1 / 2
                else:
                    raise RuntimeError("Undetermined LJ 1-4 scale")
            nbsElement.set("coulomb14scale", f"{coul14scale:.5f}")
            nbsElement.set("lj14scale", f"{lj14scale:.5f}")

            if useAtrributeFromResidue:
                useChargeElement = ET.SubElement(nbsElement, "UseAttributeFromResidue")
                useChargeElement.set("name", "charge")
            params = parmed.ParameterSet.from_structure(struct, allow_unequal_duplicates=True)
            for atom in struct.atoms:
                sigma = params.atom_types[atom.type].sigma / 10 # in nm
                #### NBFIX ####
                epsilon = params.atom_types[atom.type].epsilon * conv # in kJ
                nbElement = ET.SubElement(nbsElement, "Atom")
                nbElement.set("type", atomTypes[atom.idx])
                nbElement.set("sigma", f'{sigma:.10f}')
                nbElement.set("epsilon", f'{epsilon:.10f}')
                if not useAtrributeFromResidue:
                    nbElement.set("charge", f"{atom.charge:.10f}")
        
        xmlobj = cls()
        xmlobj.ffxml = ffxml
        xmlobj.topxml = topxml
        return xmlobj
        # return ET.ElementTree(ffxml), ET.ElementTree(topxml)


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
    neuturalize: bool = True,
    ionic_strength: float = 0.15 * unit.molar,
    gas_buffer: float = 2.0 * unit.nanometers,
    solvent_buffer: float = 1.2 * unit.nanometers,
    complex_buffer: float = 1.5 * unit.nanometers,
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
    OpenmmXML.to_pretty_xmlfile(OpenmmXML.from_parmed(ligandA_struct).ffxml, ffxml_A)
    ffxml_B = str(wdir / 'ligandB.xml')
    OpenmmXML.to_pretty_xmlfile(OpenmmXML.from_parmed(ligandB_struct).ffxml, ffxml_B)

    if protein_ff.endswith('.xml'):
        protein_ff_xml = protein_ff
    else:
        protein_ff_xml = PROTEIN_FF_XMLS[protein_ff]
    
    if water_ff.endswith('.xml'):
        water_ff_xml = water_ff
    else:
        water_ff_xml = WATER_FF_XMLS[water_ff]
    
    ff = app.ForceField(protein_ff_xml, water_ff_xml, ffxml_A, ffxml_B)

    def _save(modeller, basename):
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False)
        tmp = parmed.openmm.load_topology(modeller.topology, system, xyz=modeller.positions)
        
        # Note: In AmberTools, TIP3P water will have 3 bonds, 0 angles. If not, pmemd will raise an error:
        # `Error: Fast 3-point water residue, name and bond data incorrect!` 
        hhtype = parmed.BondType(k=553.000, req=1.514)
        for residue in tmp.residues:
            # Amber uses WAT to identify SETTLE for waters
            if residue.name == 'HOH':
                residue.name = 'WAT'
            if residue.name != 'HOH' and residue.name != 'WAT':
                continue
            atom_dict = {atom.name: atom for atom in residue.atoms}
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
    gasBoxVectors = computeBoxVectorsWithPadding(modeller.positions, gas_buffer)
    modeller.positions = shiftToBoxCenter(modeller.positions, gasBoxVectors)
    modeller.topology.setPeriodicBoxVectors(gasBoxVectors)
    _save(modeller, 'gas')
    
    # Solvent phase
    solventBoxVectors = computeBoxVectorsWithPadding(modeller.positions, solvent_buffer)
    modeller.positions = shiftToBoxCenter(modeller.positions, solventBoxVectors)
    modeller.topology.setPeriodicBoxVectors(solventBoxVectors)
    modeller.addSolvent(
        forcefield=ff,
        model=water_ff,
        neutralize=neuturalize
    )
    _save(modeller, 'solvent')

    # Complex-phase
    if protein_pdb is not None:
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
            ionicStrength=ionic_strength,
            neutralize=neuturalize
        )
        _save(modeller, 'complex')