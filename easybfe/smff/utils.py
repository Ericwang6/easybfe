from functools import wraps
import os
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET
import xml.dom.minidom
import numpy as np

import parmed
import parmed.unit as u
from parmed.periodic_table import Element as ELEMENT

import openmm as mm
import openmm.unit as unit
import openmm.app as app


def safe_join_dihedrals(struct: parmed.Structure):
    """
    Join multiple dihedral types for the same dihedral angle into a list.
    
    ParmEd cannot add two :class:`parmed.DihedralType` instances with the same
    periodicity to the same :class:`parmed.DihedralTypeList`. This function
    handles this limitation by grouping multiple dihedral types for the same
    dihedral angle into a list, which is the format expected by OpenMM XML.
    
    Parameters
    ----------
    struct : parmed.Structure
        ParmEd structure object containing dihedrals to process. The structure
        is modified in-place.
    
    Notes
    -----
    This function:
    
    1. Identifies duplicate dihedrals (same atom indices, forward or reverse)
    2. Groups their dihedral types into a list
    3. Removes duplicate dihedral entries
    4. Updates the structure's dihedral_types list
    
    The function returns early if:
    
    * No dihedral types exist
    * Dihedrals are already joined (types are lists)
    * Dihedrals are not fully parametrized (some types are None)
    
    This is a helper function used by :meth:`OpenmmXML.from_parmed` to prepare
    dihedrals for OpenMM XML conversion.
    
    See Also
    --------
    :meth:`OpenmmXML.from_parmed` : Main function that uses this helper.
    """
    # parmed can't add two DihedralType instances with the same periodicity 
    # to the same DihedralTypeList
    if not struct.dihedral_types:
        return  # nothing to do
    if any(isinstance(t, list) for t in struct.dihedral_types):
        return  # already done
    if any(d.type is None for d in struct.dihedrals):
        return  # Not fully parametrized
    dihedrals_to_delete = list()
    dihedrals_processed = dict()
    new_dihedral_types = parmed.TrackedList()
    for i, d in enumerate(struct.dihedrals):
        if d.atom1 < d.atom4:
            key = (d.atom1, d.atom2, d.atom3, d.atom4)
        else:
            key = (d.atom4, d.atom3, d.atom2, d.atom1)
        
        if key in dihedrals_processed:
            dihedrals_processed[key].append(d.type)
            dihedrals_to_delete.append(i)
        else:
            dihedrals_processed[key] = dtl = list()
            dtl.append(d.type)
            new_dihedral_types.append(dtl)
            d.type = dtl
    # Now drop the new dihedral types into place
    struct.dihedral_types = new_dihedral_types
    # Remove the "duplicate" dihedrals
    for i in reversed(dihedrals_to_delete):
        struct.dihedrals[i].delete()
        del struct.dihedrals[i]


def convert_to_xml(struct, ff_xml, top_xml=None):
    """
    Convert a ParmEd structure to OpenMM XML force field files.
    
    This is a convenience function that converts a ParmEd structure (or a file
    that can be loaded by ParmEd) into OpenMM XML format. It generates a force
    field XML file and optionally a topology XML file.
    
    Parameters
    ----------
    struct : parmed.Structure or os.PathLike
        ParmEd structure object, or path to a file that ParmEd can load
        (e.g., .prmtop, .top, .gro).
    ff_xml : os.PathLike
        Output path for the force field XML file. This file contains all
        force field parameters (bonds, angles, dihedrals, nonbonded).
    top_xml : os.PathLike, optional
        Output path for the topology XML file. This file contains residue
        definitions with atom names, types, and bonds. If None, only the
        force field XML is written.
    
    Notes
    -----
    This function is a wrapper around :class:`OpenmmXML.from_parmed` and the
    write methods. It handles loading the structure if a file path is provided.
    
    The conversion process:
    
    1. Loads the structure (if a file path is provided)
    2. Converts to OpenMM XML format using :meth:`OpenmmXML.from_parmed`
    3. Writes the force field XML file
    4. Optionally writes the topology XML file
    
    Examples
    --------
    >>> convert_to_xml('ligand.prmtop', 'ligand.xml')
    >>> convert_to_xml(struct, 'ligand.xml', 'ligand_top.xml')
    
    See Also
    --------
    :class:`OpenmmXML` : Class that performs the actual conversion.
    :meth:`OpenmmXML.from_parmed` : Method that creates XML from ParmEd structure.
    """
    if not isinstance(struct, parmed.Structure):
        struct = parmed.load_file(struct)
    obj = OpenmmXML.from_parmed(struct)
    obj.write_ffxml(ff_xml)
    if top_xml:
        obj.write_topxml(top_xml)


class OpenmmXML:
    """
    Converter from ParmEd structures to OpenMM XML force field format.
    
    This class converts ParmEd :class:`parmed.Structure` objects into OpenMM
    XML force field files. It handles conversion of all force field terms:
    bonds, angles, dihedrals (proper and improper), and nonbonded interactions.
    
    The class generates two types of XML files:
    
    * Force field XML: Contains all force field parameters (atom types, bonds,
      angles, dihedrals, nonbonded terms)
    * Topology XML: Contains residue definitions with atom names, types, and
      connectivity
    
    Attributes
    ----------
    ffxml : xml.etree.ElementTree.Element or None
        XML element tree for the force field XML file.
    topxml : xml.etree.ElementTree.Element or None
        XML element tree for the topology XML file.
    
    Examples
    --------
    >>> struct = parmed.load_file('ligand.prmtop', xyz='ligand.inpcrd')
    >>> xml_obj = OpenmmXML.from_parmed(struct)
    >>> xml_obj.write_ffxml('ligand.xml')
    >>> xml_obj.write_topxml('ligand_top.xml')
    
    See Also
    --------
    :func:`convert_to_xml` : Convenience function for conversion.
    :func:`safe_join_dihedrals` : Helper function for dihedral processing.
    """
    
    def __init__(self):
        """
        Initialize an empty OpenmmXML object.
        
        The XML elements are set by :meth:`from_parmed` class method.
        """
        self.ffxml = None
        self.topxml = None
    
    @staticmethod
    def to_pretty_xmlstr(xmlele: ET.Element):
        """
        Convert an XML element tree to a pretty-printed string.
        
        Parameters
        ----------
        xmlele : xml.etree.ElementTree.Element
            XML element tree to convert.
        
        Returns
        -------
        str
            Pretty-printed XML string with proper indentation.
        """
        uglystr = ET.tostring(xmlele, "unicode")
        pretxml = xml.dom.minidom.parseString(uglystr)
        pretstr = pretxml.toprettyxml()
        return pretstr
    
    @staticmethod
    def to_pretty_xmlfile(xmlele: ET.Element, fname: os.PathLike):
        """
        Write an XML element tree to a file with pretty printing.
        
        Parameters
        ----------
        xmlele : xml.etree.ElementTree.Element
            XML element tree to write.
        fname : os.PathLike
            Output file path.
        """
        xmlstr = OpenmmXML.to_pretty_xmlstr(xmlele)
        with open(fname, 'w') as f:
            f.write(xmlstr)
    
    def write_ffxml(self, fname: os.PathLike):
        """
        Write the force field XML to a file.
        
        Parameters
        ----------
        fname : os.PathLike
            Output file path for the force field XML.
        
        Raises
        ------
        AttributeError
            If `ffxml` is None (i.e., :meth:`from_parmed` has not been called).
        """
        self.to_pretty_xmlfile(self.ffxml, fname)
    
    def write_topxml(self, fname: os.PathLike):
        """
        Write the topology XML to a file.
        
        Parameters
        ----------
        fname : os.PathLike
            Output file path for the topology XML.
        
        Raises
        ------
        AttributeError
            If `topxml` is None (i.e., :meth:`from_parmed` has not been called).
        """
        self.to_pretty_xmlfile(self.topxml, fname)

    @classmethod
    def from_parmed(cls, struct: parmed.Structure, useAtrributeFromResidue: bool = True, onlyCharges: bool = False):
        """
        Create an OpenmmXML object from a ParmEd structure.
        
        This method converts a ParmEd structure into OpenMM XML format,
        including all force field parameters. It handles unit conversions
        from AMBER/GROMACS units to OpenMM units, and processes dihedrals
        (both proper and improper) with proper ordering schemes.
        
        Parameters
        ----------
        struct : parmed.Structure
            ParmEd structure object to convert. Must be fully parameterized.
        useAtrributeFromResidue : bool, default True
            If True, charges are stored in the residue definition and referenced
            via ``UseAttributeFromResidue`` in the nonbonded force. This allows
            the same force field to be used with different charge sets. If False,
            charges are stored directly in the nonbonded force.
        onlyCharges : bool, default False
            If True, only generate atom types and charges, skipping all bonded
            terms (bonds, angles, dihedrals). Requires `useAtrributeFromResidue`
            to be True.
        
        Returns
        -------
        OpenmmXML
            OpenmmXML object with `ffxml` and `topxml` attributes set.
        
        Raises
        ------
        AssertionError
            If `onlyCharges` is True but `useAtrributeFromResidue` is False.
        NotImplementedError
            If duplicate atom names are found in the structure.
        RuntimeError
            If improper dihedrals cannot be processed (e.g., no central atom found).
        TypeError
            If dihedral type is not recognized.
        
        Notes
        -----
        The conversion process:
        
        1. Joins dihedral types using :func:`safe_join_dihedrals`
        2. Creates unique atom names and types (format: ``{residue}-{atom}``)
        3. Generates force field XML with:
           * Atom types (element, name, class, mass)
           * Harmonic bonds (converted from :math:`\text{kcal} \cdot \text{mol}^{-1} \cdot \text{Å}^{-2}` to :math:`\text{kJ} \cdot \text{mol}^{-1} \cdot \text{nm}^{-2}`)
           * Harmonic angles (converted from :math:`\text{kcal} \cdot \text{mol}^{-1} \cdot \text{rad}^{-2}` to :math:`\text{kJ} \cdot \text{mol}^{-1} \cdot \text{rad}^{-2}`)
           * Periodic torsions (proper and improper dihedrals)
           * Nonbonded forces (LJ parameters, 1-4 scaling factors)
        4. Generates topology XML with residue definitions
        
        Unit conversions:
        
        * Bond lengths: :math:`\text{Å} \to \text{nm}` (divide by 10)
        * Bond force constants: :math:`\text{kcal} \cdot \text{mol}^{-1} \cdot \text{Å}^{-2} \to \text{kJ} \cdot \text{mol}^{-1} \cdot \text{nm}^{-2}`
        * Angle force constants: :math:`\text{kcal} \cdot \text{mol}^{-1} \cdot \text{rad}^{-2} \to \text{kJ} \cdot \text{mol}^{-1} \cdot \text{rad}^{-2}`
        * Dihedral phases: degrees :math:`\to` radians
        * Dihedral force constants: :math:`\text{kcal} \cdot \text{mol}^{-1} \to \text{kJ} \cdot \text{mol}^{-1}`
        * LJ sigma: :math:`\text{Å} \to \text{nm}` (divide by 10)
        * LJ epsilon: :math:`\text{kcal} \cdot \text{mol}^{-1} \to \text{kJ} \cdot \text{mol}^{-1}`
        
        Dihedral ordering:
        
        * Proper dihedrals use "amber" ordering by default
        * Improper dihedrals use "amber" ordering, but switch to "smirnoff"
          if all combinations of the three peripheral atoms exist
        
        1-4 scaling factors:
        
        * Extracted from GROMACS defaults section (if available) or from
          dihedral types in AMBER format
        * Supports AMBER (5/6 for coulomb, 1/2 for LJ) and OPLS (1/2 for both)
        
        Examples
        --------
        >>> struct = parmed.load_file('ligand.prmtop', xyz='ligand.inpcrd')
        >>> xml_obj = OpenmmXML.from_parmed(struct)
        >>> xml_obj.write_ffxml('ligand.xml')
        
        See Also
        --------
        :func:`safe_join_dihedrals` : Helper function for dihedral processing.
        :func:`convert_to_xml` : Convenience wrapper function.
        """
        if onlyCharges:
            assert useAtrributeFromResidue, "useAtrributeFromResidue must be True when onlyCharges is True"
        
        # need to join all dihedral types to one dihedral
        # but parmed cannot join dihedrals when two dihedral types are with 
        # the same perioidicity
        safe_join_dihedrals(struct)
        
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
                """
                Set dihedral type parameters in XML element.
                
                Parameters
                ----------
                dtype : parmed.DihedralType
                    Dihedral type to extract parameters from.
                num : int
                    Periodicity number (1, 2, 3, etc.) for multi-term dihedrals.
                diheElement : xml.etree.ElementTree.Element
                    XML element to set attributes on.
                """
                diheElement.set(f"periodicity{num}", str(dtype.per))
                diheElement.set(f"phase{num}", f"{dtype.phase * deg2rad:.10f}")
                diheElement.set(f"k{num}", f"{dtype.phi_k * conv:.10f}")
            
            def _set_dihedral(dihe: parmed.Dihedral, diheElement: ET.Element):
                """
                Set all dihedral parameters in XML element.
                
                Handles both single DihedralType and list of DihedralTypes.
                
                Parameters
                ----------
                dihe : parmed.Dihedral
                    Dihedral object containing type information.
                diheElement : xml.etree.ElementTree.Element
                    XML element to set attributes on.
                
                Raises
                ------
                TypeError
                    If dihedral type is not DihedralType or list of DihedralTypes.
                """
                if isinstance(dihe.type, parmed.DihedralType):
                    _set_dihedral_type(dihe.type, 1, diheElement)
                elif isinstance(dihe.type, list):
                    for i, dtype in enumerate(dihe.type):
                        _set_dihedral_type(dtype, i+1, diheElement)
                else:
                    raise TypeError(f"Unsupported dihedral type: {type(dihe.type)}")

            
            def _is_proper(dihe: parmed.Dihedral):
                """
                Check if a dihedral is a proper dihedral (1-2-3-4 bonded).
                
                Parameters
                ----------
                dihe : parmed.Dihedral
                    Dihedral to check.
                
                Returns
                -------
                bool
                    True if all atoms are sequentially bonded (1-2, 2-3, 3-4).
                """
                return all([
                    dihe.atom2 in dihe.atom1.bond_partners,
                    dihe.atom3 in dihe.atom2.bond_partners,
                    dihe.atom4 in dihe.atom3.bond_partners
                ])
            
            def _find_central_atom(dihe: parmed.Dihedral):
                """
                Find the central atom in an improper dihedral.
                
                The central atom is the one bonded to all three other atoms.
                This is used to determine the ordering for improper dihedrals
                in OpenMM XML format.
                
                Parameters
                ----------
                dihe : parmed.Dihedral
                    Improper dihedral to analyze.
                
                Returns
                -------
                tuple
                    (central_atom, other_atoms) where central_atom is the
                    atom bonded to all others, and other_atoms is a list of
                    the three peripheral atoms.
                
                Raises
                ------
                RuntimeError
                    If multiple central atoms are found, or if no central atom
                    is found (indicating this might be a proper dihedral).
                """
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
                """
                Determine 1-4 scaling factors from dihedral type.
                
                Extracts coulomb (scee) and van der Waals (scnb) 1-4 scaling
                factors from a dihedral. Handles both single and multi-term
                dihedrals, where some terms may have 1.0 scaling (ignored).
                
                Parameters
                ----------
                dihe : parmed.Dihedral
                    Dihedral to extract scaling factors from.
                
                Returns
                -------
                tuple
                    (scee, scnb) where scee is the coulomb 1-4 scale factor
                    and scnb is the van der Waals 1-4 scale factor.
                
                Raises
                ------
                AssertionError
                    If multiple non-1.0 scaling factors are found (inconsistent).
                
                Notes
                -----
                In prmtop files, 1.0 scaling factors are used for:
                * Dihedrals with ignore_end=True
                * Some terms in multi-periodicity torsions
                
                This function extracts the actual scaling factors, ignoring 1.0 values.
                """
                if isinstance(dihe.type, parmed.DihedralType):
                    return dihe.type.scee, dihe.type.scnb
                elif isinstance(dihe.type, list):
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
            
            for atom in struct.atoms:
                sigma = atom.sigma / 10 # in nm
                epsilon = atom.epsilon * conv # in kJ
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


def process_prmtop(func: Callable):
    """
    Decorator to validate prmtop to XML conversion by comparing energies.
    
    This decorator wraps parametrization methods to automatically:
    
    1. Convert the generated prmtop/inpcrd files to OpenMM XML format
    2. Create OpenMM systems from both the original prmtop and the XML
    3. Compare potential energies to validate the conversion
    4. Raise an error if energies differ by more than 0.01 kJ/mol
    
    The decorator is designed to be used on methods of classes that inherit
    from :class:`SmallMoleculeForceField`. It expects the wrapped method to
    generate prmtop and inpcrd files in the working directory.
    
    Parameters
    ----------
    func : Callable
        The parametrization method to wrap. Must have signature:
        ``(self, ligand_file, wdir, *args, **kwargs)``
    
    Returns
    -------
    Callable
        Wrapped function that performs validation after parametrization.
    
    Raises
    ------
    AssertionError
        If the energy difference between the prmtop and XML systems exceeds
        :math:`0.01 \, \text{kJ} \cdot \text{mol}^{-1}`, indicating an incompatible force field conversion.
    FileNotFoundError
        If required files (prmtop, inpcrd) are not found after the wrapped
        function executes.
    RuntimeError
        If OpenMM system creation or energy calculation fails.
    
    Notes
    -----
    The decorator expects the wrapped function to generate files with names
    based on the ligand file stem:
    
    * ``{stem}.prmtop``: AMBER topology file
    * ``{stem}.inpcrd``: AMBER coordinate file
    
    After the wrapped function executes, the decorator:
    
    1. Loads the prmtop/inpcrd files using ParmEd
    2. Sets residue name to 'MOL'
    3. Writes a PDB file for topology
    4. Converts to XML using :func:`convert_to_xml`
    5. Creates OpenMM systems and contexts
    6. Computes potential energies at the same coordinates
    7. Validates that energies match within tolerance
    
    Examples
    --------
    The decorator is used on parametrization methods:
    
    >>> class MyFF(SmallMoleculeForceField):
    ...     @process_prmtop
    ...     def parametrize(self, ligand_file, wdir=None):
    ...         # Generate prmtop/inpcrd files
    ...         ...
    
    See Also
    --------
    :func:`convert_to_xml` : Function that performs the XML conversion.
    :class:`SmallMoleculeForceField` : Base class for parameterizers.
    """

    @wraps(func)
    def wrapper(self, ligand_file, wdir, *args, **kwargs):
        func(self, ligand_file, wdir, *args, **kwargs)
        # Resolve paths to ensure consistency with what the function created
        ligand_file = Path(ligand_file).resolve()
        wdir = Path(wdir).resolve()
        stem = ligand_file.stem

        prmtop = str(wdir / f'{stem}.prmtop')
        inpcrd = str(wdir / f'{stem}.inpcrd')
        pdb = str(wdir / f'{stem}.pdb')
        ffxml = str(wdir / f'{stem}.xml')

        struct = parmed.load_file(prmtop, xyz=inpcrd)
        struct.residues[0].name = 'MOL'
        app.PDBFile.writeFile(struct.topology, struct.positions, pdb, keepIds=True)
        convert_to_xml(struct, ffxml)

        system_ref = app.AmberPrmtopFile(prmtop).createSystem()
        ctx_ref = mm.Context(system_ref, mm.LangevinIntegrator(300, 1.0, 0.001))
        ctx_ref.setPositions(app.AmberInpcrdFile(inpcrd).positions)
        energy_ref = ctx_ref.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        pdb_obj = app.PDBFile(pdb)
        system = app.ForceField(ffxml).createSystem(pdb_obj.topology)
        ctx = mm.Context(system, mm.LangevinIntegrator(300, 1.0, 0.001))
        ctx.setPositions(app.AmberInpcrdFile(inpcrd).positions)
        energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        assert abs(energy_ref-energy) < 0.01, \
            f"Fail to convert prmtop to xml, the force field might not be compatitable because the energy is different {energy_ref} != {energy}"
    
    return wrapper