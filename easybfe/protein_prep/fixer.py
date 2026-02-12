import os, io
from pathlib import Path
import logging
from collections import defaultdict
from typing import Optional, Dict, List, TextIO, Tuple
from textwrap import wrap

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer

logger = logging.getLogger(__name__)

aa_mapping = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR','V': 'VAL',
}

# copied from PDBFixer
proteinResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR', 'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL']
rnaResidues = ['A', 'G', 'C', 'U', 'I']
dnaResidues = ['DA', 'DG', 'DC', 'DT', 'DI']
# OpenMM amber14/tip3p.xml
ionResidues = [
    "AL", "Ag", "BA", "BR", "Be", "CA", "CD", "CE", "CL", "CO", "CR", "CS", "CU",
    "Ce", "Cr", "Dy", "EU", "EU3", "Er", "F", "FE", "FE2", "GD3", "HG", "Hf", "IN",
    "IOD", "K", "LA", "LI", "LU", "MG", "MN", "NA", "NI", "Nd", "PB", "PD", "PR",
    "PT", "Pu", "RB", "Ra", "SM", "SR", "Sm", "Sn", "TB", "Th", "Tl", "Tm",
    "U4+", "V2+", "Y", "YB2", "ZN", "Zr"
]
# Water - OpenMM will standarize WAT to HOH
waterResidues = ['HOH']
# metal residues
metalResidues = [
    "AL", "Ag", "BA", "Be", "CA", "CD", "CE", "CO", "CR", "CS", "CU",
    "Ce", "Cr", "Dy", "EU", "EU3", "Er", "FE", "FE2", "GD3", "HG", "Hf", "IN",
    "K", "LA", "LI", "LU", "MG", "MN", "NA", "NI", "Nd", "PB", "PD", "PR",
    "PT", "Pu", "RB", "Ra", "SM", "SR", "Sm", "Sn", "TB", "Th", "Tl", "Tm",
    "U4+", "V2+", "Y", "YB2", "ZN", "Zr"
]


def convert_to_three_letter_seq(sequence: str):
    """
    Convert one-letter amino acid sequence to three-letter codes.

    Handles sequences with parentheses for non-standard residues. Content
    inside parentheses is preserved as-is (e.g., for modified residues).

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence. May contain parentheses for
        non-standard residues, e.g., ``"MK(SEP)K"``.

    Returns
    -------
    List[str]
        List of three-letter residue codes. Non-standard residues in
        parentheses are kept as-is.

    Raises
    ------
    ValueError
        If unmatched parentheses are found in the sequence.

    Examples
    --------
    >>> convert_to_three_letter_seq("MK")
    ['MET', 'LYS']
    >>> convert_to_three_letter_seq("MK(SEP)K")
    ['MET', 'LYS', 'SEP', 'LYS']
    """
    # Convert the one-letter code to three-letter code
    result = []
    i = 0
    while i < len(sequence):
        if sequence[i] == '(':
            # Find the closing parenthesis
            closing_index = sequence.find(')', i)
            if closing_index != -1:
                # Append the content inside the parentheses as is
                result.append(sequence[i+1:closing_index])
                i = closing_index + 1
            else:
                raise ValueError("Unmatched parenthesis in the sequence.")
        else:
            # Convert the single-letter code to three-letter code
            result.append(aa_mapping.get(sequence[i], 'XAA'))  # 'XAA' for unknown residues
            i += 1
    return result


def convert_to_seqres(sequence: List[str], chain_id: str):
    """
    Convert a sequence list to PDB SEQRES record format.

    Creates properly formatted SEQRES lines following PDB format
    specifications, with lines wrapped at 51 characters.

    Parameters
    ----------
    sequence : List[str]
        List of three-letter residue codes.
    chain_id : str
        Single character chain identifier.

    Returns
    -------
    str
        Multi-line string containing SEQRES records, with each line
        formatted as ``"SEQRES  <num> <chain> <length>  <residues>"``.

    Examples
    --------
    >>> convert_to_seqres(['MET', 'LYS', 'GLY'], 'A')
    'SEQRES   1 A    3  MET LYS GLY'
    """
    lines = wrap(' '.join([s.upper() for s in sequence]), 51)
    # Create the SEQRES lines
    seqres_lines = []
    for i, line in enumerate(lines):
        seqres_lines.append(f"SEQRES  {i+1: >2} {chain_id} {len(sequence): >4}  {line}")
    return '\n'.join(seqres_lines)


def _residue_repr(residue):
    return f'<Residue {residue.name} {residue.id}{residue.insertionCode} (chain {residue.chain.id})>'


class ProteinFixer(PDBFixer):
    """
    Enhanced PDBFixer for fixing standardized PDB files.

    Extends :class:`PDBFixer` with additional functionality for handling
    missing residues, standardizing protonation states, and generating
    comprehensive REMARK records. Designed to work with standardized PDB
    files where:

    1. SEQRES records are well documented in the PDB header
    2. PDB residues are numbered in ``_pdbx_poly_seq_scheme.seq_id`` format
       with no insertion codes, enabling PDBFixer to find all missing residues

    Attributes
    ----------
    mod_res_info : List[Tuple[str, int, str, str, str]]
        List of modified residue information tuples:
        ``(chain_id, residue_id, insertion_code, original_name, standard_name)``
    missing_residues_added : List[Tuple[str, int, str]]
        List of added missing residues: ``(chain_id, residue_id, residue_name)``
    missing_residues_skipped : List[Tuple[str, int, str]]
        List of skipped missing residues: ``(chain_id, residue_id, residue_name)``
    missing_atoms_added : List[Tuple[str, int, str, List[str]]]
        List of added missing atoms:
        ``(chain_id, residue_id, residue_name, atom_names)``

    Notes
    -----
    The class tracks all modifications made during the fixing process,
    allowing generation of detailed REMARK records in the output PDB file.
    """
    MAX_WARN_MISSING_RES = 10
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod_res_info = []
        self.missing_residues_added = []
        self.missing_residues_skipped = []
        self.missing_atoms_added = []
    
    def standarizeResidue(self):
        """
        Standardize residue names based on protonation state and bonding.

        Determines the correct protonation state variant for residues that
        can exist in multiple forms (HIS, CYS, GLU, ASP, LYS) by examining
        hydrogen bonding patterns and metal coordination.

        The following mappings are applied:
        - ``HIS`` to ``HIP`` (both ND and NE protonated), ``HID`` (ND
          protonated), ``HIE`` (NE protonated), or ``HIN`` (neither)
        - ``CYS`` to ``CYX`` (disulfide bond), ``CYM`` (deprotonated), or
          ``CYS`` (protonated)
        - ``GLU`` to ``GLH`` (protonated) or ``GLU`` (deprotonated)
        - ``ASP`` to ``ASH`` (protonated) or ``ASP`` (deprotonated)
        - ``LYS`` to ``LYN`` (neutural) or ``LYS`` (positively charged)

        Notes
        -----
        Modifies :attr:`residue.name` in-place for residues in
        :attr:`self.topology`. Warns if ``HIN`` is detected as it's not
        supported by amber14.

        Raises
        ------
        AssertionError
            If inconsistent protonation states are detected (e.g., disulfide
            bond with hydrogen, or invalid number of hydrogens on charged
            residues).
        """
        # make bonded map
        bonded = {}
        for atom in self.topology.atoms():
            bonded[atom] = []
        for atom1, atom2 in self.topology.bonds():
            bonded[atom1].append(atom2)
            bonded[atom2].append(atom1)

        for residue in self.topology.residues():
            std_name = ''
            if residue.name == 'HIS':
                has_hd, has_he = False, False
                for at in residue.atoms():
                    if at.name.startswith('NE'):
                        has_he = any([a.element.symbol == 'H' for a in bonded[at]])
                    if at.name.startswith('ND'):
                        has_hd = any([a.element.symbol == 'H' for a in bonded[at]])
                if has_hd and has_he:
                    std_name = 'HIP'
                elif has_hd:
                    std_name = 'HID'
                elif has_he:
                    std_name = 'HIE'
                else:
                    std_name = 'HIN'
                    logger.warning('HIN is not supported by amber14')
            elif residue.name == 'CYS':
                has_h, has_s = False, False
                for at in residue.atoms():
                    if at.name.startswith("SG"):
                        has_h = any([a.element.symbol == 'H' for a in bonded[at]])
                        has_s = any([a.element.symbol == 'S' for a in bonded[at]])
                        break
                if has_s:
                    std_name = 'CYX'
                    assert not has_h, f'{residue} disulfide bond exists with S with hydrogen'
                else:
                    std_name = 'CYS' if has_h else 'CYM'
            elif residue.name == 'GLU':
                oh = 0
                for at in residue.atoms():
                    if at.name.startswith('OE'):
                        oh += sum([a.element.symbol == 'H' for a in bonded[at]])
                assert oh < 2, f'{residue} contains double hydrogen'
                std_name = 'GLU' if oh == 0 else 'GLH'
            elif residue.name == 'ASP':
                oh = 0
                for at in residue.atoms():
                    if at.name.startswith('OD'):
                        oh += sum([a.element.symbol == 'H' for a in bonded[at]])
                assert oh < 2, f'{residue} contains double hydrogen'
                std_name = 'ASP' if oh == 0 else 'ASH'
            elif residue.name == 'LYS':
                nh = 0
                for at in residue.atoms():
                    if at.name.startswith('NZ'):
                        nh += sum([a.element.symbol == 'H' for a in bonded[at]])
                assert nh == 2 or nh == 3, f'{residue} contains {nh} hydrogens on NZ, which is wrong'
                std_name = 'LYN' if nh == 2 else 'LYS'
            
            if std_name:
                residue.name = std_name
    
    def removeHeterogens(self, keep_water: bool = True, keep_ions: bool = True, extra_keep: Optional[List[str]] = None):
        """
        Remove heterogen residues from the structure.

        Removes all residues that are not standard protein, DNA, or RNA
        residues, with optional retention of water and ions. This method
        overrides the original :class:`PDBFixer.removeHeterogens` method.

        Parameters
        ----------
        keep_water : bool, optional
            If ``True``, retain water residues (default: ``True``).
        keep_ions : bool, optional
            If ``True``, retain ion residues (default: ``True``).
        extra_keep : List[str], optional
            Additional residue names to retain beyond standard residues,
            water, and ions (default: ``[]``).

        Returns
        -------
        List[Residue]
            List of residues that were deleted from the structure.

        Notes
        -----
        Modifies :attr:`self.topology` and :attr:`self.positions` in-place.
        Standard residues include those in :data:`proteinResidues`,
        :data:`dnaResidues`, and :data:`rnaResidues`.
        """
        extra_keep = list() if extra_keep is None else extra_keep
        keep = set(proteinResidues).union(dnaResidues).union(rnaResidues)
        if keep_water:
            keep = keep.union(waterResidues)
        if keep_ions:
            keep = keep.union(ionResidues)
        keep = keep.union(extra_keep)
        toDelete = []
        for residue in self.topology.residues():
            if residue.name not in keep:
                toDelete.append(residue)
        modeller = app.Modeller(self.topology, self.positions)
        modeller.delete(toDelete)
        self.topology = modeller.topology
        self.positions = modeller.positions
        return toDelete
    
    def determineVariantFromMetal(self):
        """
        Determine residue protonation variants based on metal coordination.

        Identifies residues that coordinate with metal ions and determines
        the appropriate protonation state variant. Metals within 0.3 nm of
        coordinating atoms trigger variant assignment.

        Returns
        -------
        Dict[Residue, str]
            Dictionary mapping residues to their variant names based on metal
            coordination:
            - ``HIS`` coordinating via ND -> ``HIE``
            - ``HIS`` coordinating via NE -> ``HID``
            - ``HIS`` coordinating via both -> ``HIN``
            - ``GLU`` coordinating via OE -> ``GLU``
            - ``ASP`` coordinating via OD -> ``ASP``
            - ``CYS`` coordinating via S -> ``CYX``
            - ``LYS`` coordinating via NZ -> ``LYN``

        Notes
        -----
        Only considers heavy (non-hydrogen) atoms for distance calculations.
        Uses a 0.3 nm cutoff distance for metal-coordinating atom pairs.
        """
        heavy_atoms = []
        positions_numpy = []
        for at in self.topology.atoms():
            if at.element.symbol == 'H':
                continue
            vec = self.positions[at.index]
            positions_numpy.append([vec.x, vec.y, vec.z])
            heavy_atoms.append(at)
        
        variants = {}
        for atom in self.topology.atoms():
            if atom.residue.name in metalResidues:
                v = self.positions[atom.index]
                metal_pos = np.array([v.x, v.y, v.z])
                indices = np.argwhere(np.linalg.norm(positions_numpy - metal_pos, axis=1) < 0.3).flatten()
                for index in indices:
                    hat = heavy_atoms[index]
                    hres = hat.residue
                    var = ''
                    if hres.name == 'HIS':
                        if hat.name.startswith('ND'):
                            var = 'HIE'
                        elif hat.name.startswith('NE'):
                            var = 'HID'
                        # This is rarely reached. same HIS form dative bond with two metals
                        exist_var = variants.get(hres, '')
                        if var and exist_var and (var != exist_var):
                            logger.warning(
                                "Both nitrogens on HIS%s%s (chain %s) bound with metal, variant %s will be assigned.", 
                                hres.id, hres.insertionCode, hres.chain.id, exist_var
                            )
                    elif hres.name == 'GLU':
                        var = 'GLU' if hat.name.startswith("OE") else ''
                    elif hres.name == 'ASP':
                        var = 'ASP' if hat.name.startswith("OD") else ''
                    elif hres.name == 'CYS':
                        var = 'CYX' if hat.name.startswith("S") else ''
                    elif hres.name == 'LYS':
                        var = 'LYN' if hat.name == 'NZ' else ''
                    
                    if var and hres not in variants:
                            variants[hres] = var
        
        # logging
        msg = "Found following residues bound with metal:\n"
        msg += '\n'.join([f'{_residue_repr(hres)}, assigned variant {var}' for hres, var in variants.items()])
        logger.info(msg)

        return variants
    
    def addMissingHydrogens(self, pH=7.4, forcefield=None):
        """
        Add missing hydrogen atoms to the structure.

        Adds hydrogens to all residues using OpenMM's Modeller, taking into
        account metal coordination states and pH-dependent protonation.
        After adding hydrogens, calls :meth:`standarizeResidue` to update
        residue names.

        Parameters
        ----------
        pH : float, optional
            pH value for determining protonation states (default: ``7.4``).
        forcefield : ForceField, optional
            OpenMM ForceField object. If ``None``, uses default from
            :class:`PDBFixer` (default: ``None``).

        Notes
        -----
        Modifies :attr:`self.topology` and :attr:`self.positions` in-place.
        Variants are determined by combining metal coordination states (from
        :meth:`determineVariantFromMetal`) with pH-dependent variants (from
        :meth:`_describeVariant` of the original ``PDBFixer`` class).
        """
        extraDefinitions = self._downloadNonstandardDefinitions()
        variants_metal = self.determineVariantFromMetal()
        variants = []
        for res in self.topology.residues():
            variants.append(variants_metal.get(res, self._describeVariant(res, extraDefinitions)))
        modeller = app.Modeller(self.topology, self.positions)
        modeller.addHydrogens(pH=pH, forcefield=forcefield, variants=variants, platform=self.platform)
        self.topology = modeller.topology
        self.positions = modeller.positions
        self.standarizeResidue()
                    
    def findNonstandardResidues(self):
        """
        Find and record non-standard residues in the structure.

        Calls the parent :meth:`PDBFixer.findNonstandardResidues` method and
        additionally records modification information in
        :attr:`self.mod_res_info` for later use in generating MODRES records.

        Notes
        -----
        Stores tuples of ``(chain_id, residue_id, insertion_code,
        original_name, standard_name)`` in :attr:`self.mod_res_info` for
        each non-standard residue found. Logs information about each
        non-standard residue found.
        """
        # Find non-std residues
        super().findNonstandardResidues()
        for residue, std_res_name in self.nonstandardResidues:
            self.mod_res_info.append(
                (residue.chain.id, int(residue.id), residue.insertionCode, residue.name, std_res_name)
            )
            logger.info('Found non-standard residue: %s -> %s', _residue_repr(residue), std_res_name)

    def findMissingAtoms(
        self, 
        skip_missing_terminal_residues: bool = False, 
        max_num_consecutive_missing_residues: Optional[int] = None, 
        cap_gaps: bool = True,
        force_cap_terminals: bool = False
    ):
        """
        Find missing residues and atoms in the structure.

        Combines the functionality of :meth:`PDBFixer.findMissingResidues`
        and :meth:`PDBFixer.findMissingAtoms` with enhanced control over
        terminal residue handling and gap capping.

        Parameters
        ----------
        skip_missing_terminal_residues : bool, optional
            If ``True``, skip adding missing residues at chain termini
            (default: ``True``).
        max_num_consecutive_missing_residues : int, optional
            Maximum number of consecutive missing residues to add. Gaps
            longer than this will be skipped (default: ``None``, no limit).
        cap_gaps : bool, optional
            If ``True``, add ``ACE`` and ``NME`` caps to skipped gaps
            (default: ``True``).
        force_cap_terminals : bool, optional
            If ``True``, force addition of ``ACE`` at N-terminus and
            ``NME`` at C-terminus for all chains (default: ``False``) no matter if
            that is needed.

        Notes
        -----
        Records added residues in :attr:`self.missing_residues_added`,
        skipped residues in :attr:`self.missing_residues_skipped`, and
        added atoms in :attr:`self.missing_atoms_added`. For skipped
        terminal residues without capping, adds ``OXT`` to the terminal
        residue's missing atoms list.
        """
        # delete terminal oxygen if force cap terminals
        if force_cap_terminals:
            mod = app.Modeller(self.topology, self.positions)
            ters = []
            for chain in self.topology.chains():
                for atom in list(chain.residues())[-1].atoms():
                    if atom.name == 'OXT':
                        ters.append(atom)
            mod.delete(ters)
            self.topology = mod.topology
            self.positions = mod.positions

        sequences = {seq.chainId: seq.residues for seq in self.sequences}
        chains = list(self.topology.chains())
        chain_residues = [list(chain.residues()) for chain in chains]
        
        # Find missing residues
        super().findMissingResidues()
        num_missing_residues = {chain.id: 0 for chain in chains}
        newMissingResidues = {}
        c_ter_residues = []
        for info, residues in self.missingResidues.items():
            chain_index, res_id_start = info
            chain_id = chains[chain_index].id
            is_n_ter = (res_id_start == 0) 
            is_c_ter = (num_missing_residues[chain_id] + len(residues) + res_id_start == len(sequences[chain_id]))
            is_ter = is_n_ter or is_c_ter
            is_too_long = (isinstance(max_num_consecutive_missing_residues, int) and len(residues) > max_num_consecutive_missing_residues)
            num_missing_residues[chain_id] += len(residues)

            if (is_ter and skip_missing_terminal_residues) or is_too_long:
                # This is the way PDBFixer find start index
                if res_id_start < len(chain_residues[chain_index]):
                    start = int(chain_residues[chain_index][res_id_start].id) - len(residues)
                else:
                    start = int(chain_residues[chain_index][res_id_start-1].id) + 1
                for i, residue in enumerate(residues):
                    self.missing_residues_skipped.append((chain_id, start + i, residue))
                
                residues = []

                if cap_gaps:
                    if is_n_ter:
                        residues.insert(0, 'ACE')
                    elif is_c_ter:
                        residues.append('NME')
                    elif len(residues) == 0:
                        # skipped long non-terminal missing residues
                        residues = ['NME', 'ACE']
                else:
                    # if not capped with NME, record them as terminal residues and OXT will be added
                    if not is_n_ter:
                        c_ter_residues.append(chain_residues[chain_index][res_id_start-1])
            
            if res_id_start < len(chain_residues[chain_index]):
                start = int(chain_residues[chain_index][res_id_start].id) - len(residues)
            else:
                start = int(chain_residues[chain_index][res_id_start-1].id) + 1
            for i, residue in enumerate(residues):
                self.missing_residues_added.append((chain_id, start + i, residue))
            
            if len(residues) > 0:
                newMissingResidues[info] = residues
        
        if force_cap_terminals:
            for i, chain in enumerate(chains):
                # TODO: this may be risky - I just want to skip non-protein chains. if the beginning is a non-std residue followed by normal protein
                # this will break
                if (chain_residues[i][0].name not in proteinResidues):
                    continue
                # N-terminal
                key = (chain.index, 0)
                if key not in newMissingResidues:
                    newMissingResidues[key] = ['ACE']
                elif newMissingResidues[key][0] != 'ACE':
                    newMissingResidues[key].insert(0, 'ACE')
                # C-terminal
                num_res = len(list(chain.residues()))
                key = (chain.index, num_res)
                if key not in newMissingResidues:
                    newMissingResidues[key] = ['NME']
                elif newMissingResidues[key][-1] != 'NME':
                    newMissingResidues[key].append('NME')

        # Check if number of missing residues (excluding ACE, NME) exceeds warning threshold
        print(newMissingResidues)
        for info, residues in newMissingResidues.items():
            # Count only non-cap residues
            num_miss = len([r for r in residues if r not in ('ACE', 'NME')])
        
            if num_miss > self.MAX_WARN_MISSING_RES:
                logger.warning(
                    'Too many missing residues (%d, exceed %d) starting from residue %s%s in chain %s. '
                    'The resulting structure maybe not trustworthy.',
                    num_miss, self.MAX_WARN_MISSING_RES,
                    chain_residues[info[0]][info[1]].id,
                    chain_residues[info[0]][info[1]].insertionCode,
                    chains[info[0]].id,
                )

        self.missingResidues = newMissingResidues
        
        # Find Missing Atoms
        super().findMissingAtoms()
        for ter in c_ter_residues:
            if ter not in self.missingTerminals:
                self.missingTerminals[ter] = ['OXT']
        if force_cap_terminals:
            self.missingTerminals = {}
        
        missingAtomNames = {residue: [at.name for at in atoms] for residue, atoms in self.missingAtoms.items()}
        for residue in self.missingTerminals:
            if residue not in missingAtomNames:
                missingAtomNames[residue] = self.missingTerminals[residue]
            else:
                missingAtomNames[residue] += self.missingTerminals[residue]

        for residue, atom_names in missingAtomNames.items():            
            self.missing_atoms_added.append(
                (residue.chain.id, int(residue.id), residue.name, atom_names)
            )

    def addMissingAtoms(self, *args, **kwargs):
        """
        Add missing atoms and residues to the structure.

        Calls the parent :meth:`PDBFixer.addMissingAtoms` method and then
        corrects residue numbering for ``NME`` caps that were incorrectly
        numbered by PDBFixer in non-fixed loop regions.

        Parameters
        ----------
        *args
            Positional arguments passed to :meth:`PDBFixer.addMissingAtoms`.
        **kwargs
            Keyword arguments passed to :meth:`PDBFixer.addMissingAtoms`.

        Notes
        -----
        Fixes a PDBFixer bug where ``NME`` residues added to cap skipped
        gaps are numbered incorrectly. For example, if residues 10-18 are
        missing and skipped, PDBFixer may number the cap as residue 17,
        but it should be numbered as residue 10. This method corrects such
        numbering issues.

        After completion, logs the lists of added residues, skipped
        residues, and added atoms.
        """
        super().addMissingAtoms(*args, **kwargs)
        # Due to the way PDBFixer number missing residues, the NME residue in non-fixed loops will be wrong in numbering
        # For example, if residues 10-18 are missing and they are not added
        # the added cap NME and ACE will be #17 and #18. However, the NME should be #10.
        prev_residue = None
        for chain in self.topology.chains():
            for residue in chain.residues():
                if residue.name == 'NME':
                    d = (chain.id, int(residue.id), residue.name)
                    if d in self.missing_residues_added:
                        index = self.missing_residues_added.index(d)
                        self.missing_residues_added[index] = (chain.id, int(prev_residue.id) + 1, residue.name)
                        residue.id = str(int(prev_residue.id) + 1)
                prev_residue = residue
        
        # Logging
        if self.missing_residues_added:
            logger.info('Add  missing residues:\n %s', self.missing_residues_added)
        if self.missing_residues_skipped:
            logger.info('Skip missing residues:\n %s', self.missing_residues_skipped)
        if self.missing_atoms_added:
            logger.info('Add  missing atoms:\n %s', self.missing_atoms_added)

    @classmethod
    def _getMappedResnumWithIcode(cls, res_id: int, chain: str, res_num_mapping: Dict[str, Dict[int, str]]) -> Tuple[str, str]:
        res_id_mapped = res_num_mapping[chain][res_id]
        insert_code = ' '
        if res_id_mapped[-1].isalpha():
            insert_code = res_id_mapped[-1]
            res_id_mapped = res_id_mapped[:-1]
        return res_id_mapped, insert_code

    @classmethod
    def getFixedResidueRemarks(cls, missing_residues_info: List[Tuple[str, int, str]], res_num_mapping=None, use_fixed_remark: bool = True):
        if len(missing_residues_info) == 0:
            return []
        
        remark_465_lines = [
            "REMARK 465",
            "REMARK 465 FIXED MISSING RESIDUES" if use_fixed_remark else  "REMARK 465 MISSING RESIDUES",
            "REMARK 465 (M=MODEL NUMBER; RES=RESIDUE NAME; C=CHAIN",
            "REMARK 465 IDENTIFIER; SSSEQ=SEQUENCE NUMBER; I=INSERTION CODE.)",
            "REMARK 465",
            "REMARK 465   M RES C SSSEQI"
        ]
        for chain, res_id, res_name in missing_residues_info:
            if not res_num_mapping:
                insert_code = ' '
            else:
                res_id, insert_code = cls._getMappedResnumWithIcode(res_id, chain, res_num_mapping)
            remark_465_lines.append(f"REMARK 465     {res_name:3} {chain:1}  {res_id:>4}{insert_code:>1}")
        return remark_465_lines
    
    @classmethod
    def getFixedAtomRemarks(cls, missing_atoms_info: List[Tuple[str, str, int, List[str]]], res_num_mapping=None):
        if len(missing_atoms_info) == 0:
            return []
        
        remark_470_lines = [
            "REMARK 470",
            "REMARK 470 FIXED MISSING ATOM",
            "REMARK 470 (M=MODEL NUMBER; RES=RESIDUE NAME; C=CHAIN",
            "REMARK 470 IDENTIFIER; SSSEQ=SEQUENCE NUMBER; I=INSERTION CODE.)",
            "REMARK 470",
            "REMARK 470   M RES CSSEQI  ATOMS"
        ]
        for chain, res_id, res_name, atoms in missing_atoms_info:
            if not res_num_mapping:
                insert_code = ' '
            else:
                res_id, insert_code = cls._getMappedResnumWithIcode(res_id, chain, res_num_mapping)
            atom_list = " ".join([f"{atom:<4}" for atom in atoms])
            remark_470_lines.append(f"REMARK 470     {res_name:3} {chain:1} {res_id:>4}{insert_code:>1}  {atom_list}")
        return remark_470_lines
    
    @classmethod
    def getModresRecords(cls, mod_res_info: List[Tuple[str, int, str, str, str]], res_num_mapping=None, pdb_id=""):
        pdb_id = 'XXXX' if not pdb_id else pdb_id.upper()
        modres_lines = []
        for chain, res_id, icode, res_name, std_res_name in mod_res_info:
            if res_num_mapping:
                res_id = res_num_mapping[chain][res_id]
                if res_id[-1].isalpha():
                    icode = res_id[-1]
                    res_id = res_id[:-1]
            # TODO: need to figure out how to deal with modfied residues with insertion code (Eric)
            modres_lines.append(f"MODRES {pdb_id:>4} {res_name:3} {chain:1} {res_id:>4}{icode:1} {std_res_name:3}   MODIFIED RESIDUE")
        return modres_lines
    
    def refineAddedAtomPositions(self, forcefield=None):
        """
        Refine positions of added atoms using energy minimization.

        Performs energy minimization to optimize the positions of atoms
        that were added to the structure (missing atoms and residues).
        Original atoms are constrained during minimization, while only
        added atoms are allowed to move.

        Parameters
        ----------
        forcefield : ForceField, optional
            OpenMM ForceField object. If ``None``, uses amber14-all.xml
            with tip3p water model (default: ``None``).

        Returns
        -------
        Positions
            Refined atomic positions after energy minimization.

        Notes
        -----
        Constrains all atoms except:
        - Atoms in added residues (from :attr:`self.missing_residues_added`)
        - Atoms that were added to existing residues (from
          :attr:`self.missing_atoms_added`)
        - Hydrogen atoms
        - Atoms in non-standard residues (entire residue constrained)

        Uses Langevin dynamics at 300 K with 10 ps^-1 friction and
        5 fs timestep for minimization with tolerance of 10.
        """
        if forcefield is None:
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
        # Conver List missing atoms information to dictionary, for better indexing
        missing_atoms_info_as_dict = defaultdict(list)
        for chain, res_id, res_name, atoms in self.missing_atoms_added:
            missing_atoms_info_as_dict[(chain, res_id, res_name)] += atoms
        system = forcefield.createSystem(self.topology, nonbondedMethod=app.CutoffNonPeriodic, constraints=None, rigidWater=False)
        nonstd_names = [res.name for res, stdname in self.nonstandardResidues]
        for residue in self.topology.residues():
            resdata = (residue.chain.id, int(residue.id), residue.name)
            if resdata in self.missing_residues_added:
                logger.debug('Found fixed residue: %s', residue)
                continue
            
            for i, atom in enumerate(residue.atoms()):
                # Always constrained all atoms in modified residue, including hydrogens (because we don't have good force field)
                if residue.name in nonstd_names:
                    system.setParticleMass(atom.index, 0.0)
                    continue
                if (resdata in missing_atoms_info_as_dict) and (atom.name in missing_atoms_info_as_dict[resdata]):
                    logger.debug('Found fixed atom: %s', atom)
                    continue
                if atom.element is app.element.hydrogen:
                    continue
                # if (self._has_ligand) and (residue.index == self.topology.getNumResidues() - 1) and (i in self.ligand_missing_atoms):
                #     continue
                system.setParticleMass(atom.index, 0.0)

        integrator = mm.LangevinIntegrator(300*unit.kelvin, 10/unit.picosecond, 5*unit.femtosecond)
        context = mm.Context(system, integrator)
        context.setPositions(self.positions)
        mm.LocalEnergyMinimizer.minimize(context, tolerance=10)
        self.positions = context.getState(getPositions=True).getPositions()
        return self.positions
    
    def run(
        self,
        skip_missing_terminal_residues: bool = False,
        max_num_consecutive_missing_residues: Optional[int] = None,
        keep_water: bool = True,
        keep_ions: bool = True,
        extra_keep: List[str] = None,
        pH: float = 7.4,
        forcefield = None,
        cap_gaps: bool = True,
        force_cap_terminals: bool = False,
        res_num_mapping: Optional[Dict[str, Dict[int, str]]] = None,
        out: str | Path | TextIO | None = None
    ) -> str | None:
        """
        Run the complete protein fixing pipeline.

        Executes the full workflow to fix a PDB structure: finding and
        replacing non-standard residues, removing heterogens, finding and
        adding missing atoms/residues, adding hydrogens, refining positions,
        and writing the output file with appropriate REMARK records.

        Parameters
        ----------
        output_protein : str
            Path to output PDB file where the fixed structure will be saved.
        res_num_mapping : Dict[str, Dict[int, str]], optional
            Mapping from chain ID to residue number mapping. Used to restore
            original PDB residue numbering and insertion codes in the output
            (default: ``None``).
        skip_missing_terminal_residues : bool, optional
            If ``True``, skip adding missing residues at chain termini
            (default: ``False``). See :meth:`findMissingAtoms` for details.
        max_num_consecutive_missing_residues : int, optional
            Maximum number of consecutive missing residues to add. If a gap
            contains more than this number of consecutive missing residues,
            the entire gap will be skipped (not added). Skipped gaps may be
            capped with ``ACE`` and ``NME`` if ``cap_gaps=True``
            (default: ``None``, meaning it tries to fix all missing residues
            regardless of gap length). See :meth:`findMissingAtoms` for details.
        keep_water : bool, optional
            If ``True``, retain water residues when removing heterogens
            (default: ``True``). See :meth:`removeHeterogens` for details.
        keep_ions : bool, optional
            If ``True``, retain ion residues when removing heterogens
            (default: ``True``). See :meth:`removeHeterogens` for details.
        extra_keep : List[str], optional
            Additional residue names to retain when removing heterogens
            (default: ``None``). See :meth:`removeHeterogens` for details.
        pH : float, optional
            pH value for determining protonation states when adding
            hydrogens (default: ``7.4``). See :meth:`addMissingHydrogens`
            for details.
        forcefield : ForceField, optional
            OpenMM ForceField object for adding hydrogens and refining
            positions. If ``None``, uses default behavior:
            
            - For adding hydrogens: uses the default forcefield from
              :class:`PDBFixer`. See :meth:`addMissingHydrogens` for details.
            - For refining positions: uses ``amber14-all.xml`` with
              ``amber14/tip3p.xml`` water model. See
              :meth:`refineAddedAtomPositions` for details.
              
            (default: ``None``).
        cap_gaps : bool, optional
            If ``True``, add ``ACE`` and ``NME`` caps to skipped gaps
            (default: ``True``). See :meth:`findMissingAtoms` for details.
        force_cap_terminals : bool, optional
            If ``True``, force addition of ``ACE`` at N-terminus and
            ``NME`` at C-terminus for all chains (default: ``False``) no matter if
            that is needed. See :meth:`findMissingAtoms` for details.

        Notes
        -----
        The workflow consists of:
        1. Finding and replacing non-standard residues
        2. Removing heterogens (with optional water/ion retention)
        3. Finding missing atoms and residues
        4. Adding missing atoms and residues
        5. Adding missing hydrogens
        6. Refining positions of added atoms
        7. Writing output PDB with REMARK 465 (missing residues), REMARK 470
           (missing atoms), and MODRES records

        If ``res_num_mapping`` is provided, residue numbers and insertion
        codes are restored to their original PDB values in the output file.
        """
        
        self.findNonstandardResidues()
        self.replaceNonstandardResidues()
        self.removeHeterogens(keep_water=keep_water, keep_ions=keep_ions, extra_keep=extra_keep)
        self.findMissingAtoms(
            skip_missing_terminal_residues=skip_missing_terminal_residues,
            max_num_consecutive_missing_residues=max_num_consecutive_missing_residues,
            cap_gaps=cap_gaps,
            force_cap_terminals=force_cap_terminals
        )
        self.addMissingAtoms()
        self.addMissingHydrogens(pH=pH, forcefield=forcefield)
        self.refineAddedAtomPositions(forcefield=forcefield)

        protein_top = self.topology
        protein_pos = self.positions
        
        # Save protein
        seqres = [(seq.chainId, convert_to_seqres(seq.residues, seq.chainId)) for seq in self.sequences]
        seqres.sort(key=lambda x: x[0])
        headers = [x[1] for x in seqres]
        headers += ProteinFixer.getFixedResidueRemarks(self.missing_residues_skipped, res_num_mapping, use_fixed_remark=False)
        headers += ProteinFixer.getFixedResidueRemarks(self.missing_residues_added, res_num_mapping)
        headers += ProteinFixer.getFixedAtomRemarks(self.missing_atoms_added, res_num_mapping)
        headers += ProteinFixer.getModresRecords(self.mod_res_info, res_num_mapping, '')

        if out is not None:
            is_file = (not hasattr(out, 'write'))
            fp = open(out, 'w') if is_file else out
        else:
            is_file = False
            fp = io.StringIO()

        app.PDBFile.writeHeader(protein_top, fp)
        for line in headers:
            print(line, file=fp)
        # map back residue id and icode
        if res_num_mapping:
            for residue in protein_top.residues():
                res_id = res_num_mapping[residue.chain.id][int(residue.id)]
                if res_id[-1].isalpha():
                    insert_code = res_id[-1]
                    res_id = res_id[:-1]
                else:
                    insert_code = " "
                residue.id = res_id
                residue.insertionCode = insert_code

        app.PDBFile.writeModel(protein_top, protein_pos, keepIds=True, file=fp)
        app.PDBFile.writeFooter(protein_top, file=fp)

        if is_file:
            fp.close()
        if out is None:
            fp.seek(0)
            return fp.read()
        return
        
            
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    fixer = ProteinFixer(
        # "/Users/ericwang/Documents/ReasonedTherapeutics/chd1/8umg_chain_A_water.pdb"
        # '/Users/ericwang/Documents/Berkeley/easybfe/tests/data/proteinfixer/5ROB.pdb',
        '/Users/ericwang/Documents/Berkeley/cache/PGK1_cmp45_water.pdb'

    )
    fixer.run(
        skip_missing_terminal_residues=False, 
        force_cap_terminals=False, 
        out='/Users/ericwang/Documents/Berkeley/cache/PGK1_cmp45_water_prepared.pdb',
        keep_ions=False,
        keep_water=True
    )
