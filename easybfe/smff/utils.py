import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
import numpy as np

import parmed
import parmed.unit as u
from parmed.periodic_table import Element as ELEMENT



def convert_to_xml(struct, ff_xml):
    if not isinstance(struct, parmed.Structure):
        struct = parmed.load_file(struct)
    OpenmmXML.from_parmed(struct).write_ffxml(ff_xml)


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
            def _set_dihedral_type(dtype, num, diheElement):
                diheElement.set(f"periodicity{num}", str(dtype.per))
                diheElement.set(f"phase{num}", f"{dtype.phase * deg2rad:.10f}")
                diheElement.set(f"k{num}", f"{dtype.phi_k * conv:.10f}")
            
            def _set_dihedral(dihe: parmed.Dihedral, diheElement: ET.Element):
                if isinstance(dihe.type, parmed.DihedralType):
                    _set_dihedral_type(dihe.type, 1, diheElement)
                elif isinstance(dihe.type, parmed.DihedralTypeList):
                    for i, dtype in enumerate(dihe.type):
                        _set_dihedral_type(dtype, i+1, diheElement)
                else:
                    raise TypeError(f"Unsupported dihedral type: {type(dihe.type)}")

            
            def _is_proper(dihe: parmed.Dihedral):
                return all([
                    dihe.atom2 in dihe.atom1.bond_partners,
                    dihe.atom3 in dihe.atom2.bond_partners,
                    dihe.atom4 in dihe.atom3.bond_partners
                ])
            
            def _find_central_atom(dihe: parmed.Dihedral):
                dihe_atoms = [dihe.atom1, dihe.atom2, dihe.atom3, dihe.atom4]
                central_atoms = []
                for i, atom in enumerate(dihe_atoms):
                    other_atoms = [dihe_atoms[k] for k in range(4) if i != k]
                    if all(at in atom.bond_partners for at in other_atoms):
                        central_atoms.append(atom)
                if len(central_atoms) > 1:
                    raise RuntimeError(f'Multiple central atoms are found: {dihe}')
                elif len(central_atoms) == 0:
                    raise RuntimeError(f"No central atoms {dihe}. Is this a proper dihedral?")
                central_atom = central_atoms[0]
                other_atoms = [at for at in dihe_atoms if at is not central_atom]
                return central_atoms[0], other_atoms
            
            def _determine_14scales(dihe: parmed.Dihedral):
                if isinstance(dihe.type, parmed.DihedralType):
                    return dihe.type.scee, dihe.type.scnb
                elif isinstance(dihe.type, parmed.DihedralTypeList):
                    # In prmtop file, sometimes the scee/scnb will be tagged to 1.0 for ignore_end=True
                    # And sometimes, for a torsion with more than one periodicity,
                    # one periodicity type will have scee=1.0 & scnb=1.0
                    scees, scnbs = [], []
                    scee, scnb = None, None
                    for dtype in dihe.type:
                        scees.append(dtype.scee)
                        scnbs.append(dtype.scnb)

                    if all(value == 1.0 for value in scees):
                        scee = 1.0
                    else:
                        scee_non1 = [value for value in scees if value != 1.0]
                        assert np.allclose(scee_non1, scee_non1[0]), "Multiple coul 1-4 scales"
                        scee = scee_non1[0]
                    
                    if all(value == 1.0 for value in scnbs):
                        scnb = 1.0
                    else:
                        scnb_non1 = [value for value in scnbs if value != 1.0]
                        assert np.allclose(scnb_non1, scnb_non1[0]), "Multiple coul 1-4 scales"
                        scnb = scnb_non1[0]
                    return scee, scnb
            

            dihesElement = ET.SubElement(ffxml, "PeriodicTorsionForce")
            # Default use amber ordering scheme, in which only one improper is added around one central atom
            dihesElement.set("ordering", "amber")
            conv = u.kilocalories.conversion_factor_to(u.kilojoules)

            proper_dihes, improper_dihes = [], []
            for dihe in struct.dihedrals:
                if _is_proper(dihe):
                    proper_dihes.append(dihe)
                else:
                    improper_dihes.append(dihe)
            
            for dihe in proper_dihes:
                diheElement = ET.SubElement(dihesElement, "Proper")
                diheElement.set("type1", str(atomTypes[dihe.atom1.idx]))
                diheElement.set("type2", str(atomTypes[dihe.atom2.idx]))
                diheElement.set("type3", str(atomTypes[dihe.atom3.idx]))
                diheElement.set("type4", str(atomTypes[dihe.atom4.idx]))
                _set_dihedral(dihe, diheElement)
            
            improper_dihes_atoms = [(dihe.atom1.idx, dihe.atom2.idx, dihe.atom3.idx, dihe.atom4.idx) for dihe in improper_dihes]
            for dihe in improper_dihes:
                diheElement = ET.SubElement(dihesElement, "Improper")
                central_atom, other_atoms = _find_central_atom(dihe)
                diheElement.set("type1", str(atomTypes[central_atom.idx]))
                diheElement.set("type2", str(atomTypes[other_atoms[0].idx]))
                diheElement.set("type3", str(atomTypes[other_atoms[1].idx]))
                diheElement.set("type4", str(atomTypes[other_atoms[2].idx]))
                _set_dihedral(dihe, diheElement)
                # All combinations exist, the ordering is smniorff
                if diheElement.get('ordering') != 'smirnoff':
                    indices = [at.idx for at in other_atoms]
                    if (central_atom.idx, indices[1], indices[2], indices[0]) in improper_dihes_atoms and \
                          (central_atom.idx, indices[2], indices[0], indices[1]) in improper_dihes_atoms:
                        dihesElement.set('ordering', 'smirnoff')
            
            # nonbonded terms
            nbsElement = ET.SubElement(ffxml, "NonbondedForce")
            if hasattr(struct, "defaults"):
                # TODO: 1-4 factors should be deduced from pair if gen_pair=='no' in gmx 
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
                scees, scnbs = [], []
                for dihe in struct.dihedrals:
                    if dihe.ignore_end or dihe.improper:
                        continue
                    scee, scnb = _determine_14scales(dihe)
                    scees.append(scee)
                    scnbs.append(scnb)
                
                assert np.allclose(scees, scees[0]), 'More than one coul 1-4 scales'
                assert np.allclose(scnbs, scnbs[0]), 'More than one vdw 1-4 scales'
                coul14scale = 1 / scees[0]
                lj14scale = 1 / scnbs[0]
            
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