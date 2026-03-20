import os, io
from pathlib import Path
import logging
from collections import defaultdict
from typing import Optional, Dict, List, TextIO, Tuple


import numpy as np
from scipy.spatial.distance import cdist
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer

from .utils import *
from .utils import _residue_repr

logger = logging.getLogger(__name__)


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
        self.wizard = kwargs.pop("wizard", False)
        super().__init__(*args, **kwargs)
        self.mod_res_info = []
        self.missing_residues_added = []
        self.missing_residues_skipped = []
        self.missing_atoms_added = []
    
    def msg(self, msg):
        if not self.wizard:
            logger.info(msg)
        else:
            print(msg)
    
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
    
    def _filter_by_distance(self, keep_residues, candidate_residues, distance):
        """Return candidate residues within ``distance`` Angstrom of any kept residue."""
        positions_angstrom = np.array(self.positions.value_in_unit(unit.angstroms))
        keep_atom_indices = np.array(
            [a.index for res in keep_residues for a in res.atoms()],
            dtype=np.intp,
        )
        if len(keep_atom_indices) == 0:
            return []
        pos_keep = positions_angstrom[keep_atom_indices]
        candidate_atom_indices = []
        residue_slices = []
        for res in candidate_residues:
            start = len(candidate_atom_indices)
            candidate_atom_indices.extend(a.index for a in res.atoms())
            residue_slices.append((start, len(candidate_atom_indices), res))
        if not candidate_atom_indices:
            return []
        pos_candidates = positions_angstrom[np.array(candidate_atom_indices, dtype=np.intp)]
        dists = cdist(pos_candidates, pos_keep)
        min_dist_per_candidate = np.min(dists, axis=1)
        result = []
        for start, end, res in residue_slices:
            if np.any(min_dist_per_candidate[start:end] < distance):
                result.append(res)
        return result
    
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
        self.msg(msg)

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
                    
    def fixNonStandardResidues(self, replace_nonstandard: bool = True):
        """
        Find non-standard residues, optionally replace them.

        Calls the parent :meth:`PDBFixer.findNonstandardResidues` to detect
        non-standard residues. In wizard mode, prompts the user to replace
        all, replace none, or manually select. In non-wizard mode, uses the
        ``replace_nonstandard`` parameter. Records modification info in
        :attr:`self.mod_res_info` for residues that are replaced, then calls
        :meth:`replaceNonstandardResidues`.

        Parameters
        ----------
        replace_nonstandard : bool, optional
            If ``True``, replace all non-standard residues with their standard
            equivalents (default: ``True``). Only used in non-wizard mode.
        """
        super().findNonstandardResidues()
        to_replace = list(self.nonstandardResidues)

        if to_replace:
            if self.wizard:
                self.msg("")
                self.msg("  Non-standard residues")
                self.msg("  " + "=" * 52)
                for i, (residue, std_name) in enumerate(to_replace):
                    self.msg(
                        "  [%d]  %s  ->  %s  (chain %s, res %s%s)"
                        % (i, residue.name, std_name, residue.chain.id, residue.id, residue.insertionCode or "")
                    )
                self.msg("  " + "=" * 52)
                while True:
                    option = input(
                        "    1 Replace all    2 Replace none    3 Choose per residue\n"
                        "  Choice [1-3]: "
                    ).strip()
                    if option == "1":
                        break
                    if option == "2":
                        to_replace = []
                        break
                    if option == "3":
                        to_replace = []
                        for residue, std_name in self.nonstandardResidues:
                            while True:
                                ans = input(
                                    "  Replace %s (chain %s, res %s%s) -> %s? [Y/n]: "
                                    % (residue.name, residue.chain.id, residue.id, residue.insertionCode or "", std_name)
                                ).strip()
                                if ans.lower() == "n":
                                    break
                                if ans.lower() == "y" or ans == "":
                                    to_replace.append((residue, std_name))
                                    break
                                print("  -> Enter y or n")
                        break
                    print("  -> Enter 1, 2, or 3")
            elif not replace_nonstandard:
                to_replace = []

        self.nonstandardResidues = to_replace

        for residue, std_res_name in self.nonstandardResidues:
            self.mod_res_info.append(
                (residue.chain.id, int(residue.id), residue.insertionCode, residue.name, std_res_name)
            )
            self.msg(f"Replace non-standard residue: {_residue_repr(residue)} -> {std_res_name}")

        if self.nonstandardResidues:
            self.replaceNonstandardResidues()

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

        In wizard mode, prompts the user per gap (fix / skip / skip and cap)
        and per chain terminal (cap N-terminus / C-terminus). In non-wizard
        mode, uses the provided parameters.

        Parameters
        ----------
        skip_missing_terminal_residues : bool, optional
            If ``True``, skip adding missing residues at chain termini
            (default: ``False``). Only used in non-wizard mode.
        max_num_consecutive_missing_residues : int, optional
            Maximum number of consecutive missing residues to add. Gaps
            longer than this will be skipped (default: ``None``, no limit).
            Only used in non-wizard mode.
        cap_gaps : bool, optional
            If ``True``, add ``ACE`` and ``NME`` caps to skipped gaps
            (default: ``True``). Only used in non-wizard mode.
        force_cap_terminals : bool, optional
            If ``True``, force addition of ``ACE`` at N-terminus and
            ``NME`` at C-terminus for all chains (default: ``False``).
            Only used in non-wizard mode.

        Notes
        -----
        Records added residues in :attr:`self.missing_residues_added`,
        skipped residues in :attr:`self.missing_residues_skipped`, and
        added atoms in :attr:`self.missing_atoms_added`. For skipped
        terminal residues without capping, adds ``OXT`` to the terminal
        residue's missing atoms list.
        """
        sequences = {seq.chainId: seq.residues for seq in self.sequences}
        chains = list(self.topology.chains())
        chain_residues = [list(chain.residues()) for chain in chains]

        # --- Determine which chain terminals to cap ---
        # cap_n / cap_c map chain ID -> bool
        cap_n = {}
        cap_c = {}
        if self.wizard:
            self.msg("")
            self.msg("  Terminal capping")
            self.msg("  " + "=" * 52)
            for chain in chains:
                has_protein = any(r.name in proteinResidues for r in chain.residues())
                if not has_protein:
                    continue
                while True:
                    ans = input(f"  Cap N-terminus of chain {chain.id}? [y/N]: ").strip()
                    if ans.lower() == 'y':
                        cap_n[chain.id] = True
                        break
                    if ans.lower() == 'n' or ans == '':
                        cap_n[chain.id] = False
                        break
                    print("  -> Enter y or n")
                while True:
                    ans = input(f"  Cap C-terminus of chain {chain.id}? [y/N]: ").strip()
                    if ans.lower() == 'y':
                        cap_c[chain.id] = True
                        break
                    if ans.lower() == 'n' or ans == '':
                        cap_c[chain.id] = False
                        break
                    print("  -> Enter y or n")
        else:
            for chain in chains:
                cap_n[chain.id] = force_cap_terminals
                cap_c[chain.id] = force_cap_terminals

        # Remove OXT from chains whose C-terminus will be capped
        any_cap_c = any(cap_c.get(c.id, False) for c in chains)
        if any_cap_c:
            mod = app.Modeller(self.topology, self.positions)
            ters = []
            for chain in self.topology.chains():
                if not cap_c.get(chain.id, False):
                    continue
                for atom in list(chain.residues())[-1].atoms():
                    if atom.name == 'OXT':
                        ters.append(atom)
            if ters:
                mod.delete(ters)
                self.topology = mod.topology
                self.positions = mod.positions
                chains = list(self.topology.chains())
                chain_residues = [list(chain.residues()) for chain in chains]

        # --- Find missing residues ---
        super().findMissingResidues()
        num_missing_residues = {chain.id: 0 for chain in chains}
        newMissingResidues = {}
        c_ter_residues = []

        if self.wizard and self.missingResidues:
            self.msg("")
            self.msg("  Missing residues")
            self.msg("  " + "=" * 52)

        for info, residues in self.missingResidues.items():
            chain_index, res_id_start = info
            chain_id = chains[chain_index].id
            is_n_ter = (res_id_start == 0)
            is_c_ter = (num_missing_residues[chain_id] + len(residues) + res_id_start == len(sequences[chain_id]))
            is_ter = is_n_ter or is_c_ter
            is_too_long = (isinstance(max_num_consecutive_missing_residues, int) and len(residues) > max_num_consecutive_missing_residues)
            num_missing_residues[chain_id] += len(residues)

            # Compute gap location for display / recording
            if res_id_start < len(chain_residues[chain_index]):
                gap_start = int(chain_residues[chain_index][res_id_start].id) - len(residues)
            else:
                gap_start = int(chain_residues[chain_index][res_id_start - 1].id) + 1

            # --- Decide: fix, skip, or skip_and_cap ---
            if self.wizard:
                gap_type = "N-terminal" if is_n_ter else ("C-terminal" if is_c_ter else "loop")
                gap_end = gap_start + len(residues) - 1
                self.msg(f"  Missing {gap_type} residues {gap_start}-{gap_end} in chain {chain_id} ({len(residues)} residues)")
                if len(residues) > self.MAX_WARN_MISSING_RES:
                    self.msg(f"  WARNING: Large gap ({len(residues)} residues) - result may not be trustworthy")
                while True:
                    option = input("    1 Fix    2 Skip    3 Skip and cap\n  Choice [1-3]: ").strip()
                    if option in ('1', '2', '3'):
                        break
                    print("  -> Enter 1, 2, or 3")
                should_skip = option in ('2', '3')
                should_cap = option == '3'
            else:
                should_skip = (is_ter and skip_missing_terminal_residues) or is_too_long
                should_cap = cap_gaps if should_skip else False

            if should_skip:
                for i, residue in enumerate(residues):
                    self.missing_residues_skipped.append((chain_id, gap_start + i, residue))

                residues = []
                if should_cap:
                    if is_n_ter:
                        residues.insert(0, 'ACE')
                    elif is_c_ter:
                        residues.append('NME')
                    elif len(residues) == 0:
                        residues = ['NME', 'ACE']
                else:
                    if not is_n_ter:
                        c_ter_residues.append(chain_residues[chain_index][res_id_start - 1])

            # Record added residues (original gap or caps)
            if res_id_start < len(chain_residues[chain_index]):
                start = int(chain_residues[chain_index][res_id_start].id) - len(residues)
            else:
                start = int(chain_residues[chain_index][res_id_start - 1].id) + 1
            for i, residue in enumerate(residues):
                self.missing_residues_added.append((chain_id, start + i, residue))

            if len(residues) > 0:
                newMissingResidues[info] = residues

        # --- Apply terminal capping (ACE / NME) ---
        for i, chain in enumerate(chains):
            protein_indices = [
                j for j, r in enumerate(chain_residues[i])
                if r.name in proteinResidues
            ]
            if not protein_indices:
                continue
            first_protein = protein_indices[0]
            last_protein = protein_indices[-1]
            if cap_n.get(chain.id, False):
                key = (chain.index, first_protein)
                if key not in newMissingResidues:
                    newMissingResidues[key] = ['ACE']
                elif newMissingResidues[key][0] != 'ACE':
                    newMissingResidues[key].insert(0, 'ACE')
            if cap_c.get(chain.id, False):
                key = (chain.index, last_protein + 1)
                if key not in newMissingResidues:
                    newMissingResidues[key] = ['NME']
                elif newMissingResidues[key][-1] != 'NME':
                    newMissingResidues[key].append('NME')

        # Warn about large gaps
        for info, residues in newMissingResidues.items():
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

        # --- Find missing atoms ---
        super().findMissingAtoms()
        for ter in c_ter_residues:
            if ter not in self.missingTerminals:
                self.missingTerminals[ter] = ['OXT']

        # Clear terminals for capped chains
        capped_chain_ids = set()
        for cid, do_cap in cap_n.items():
            if do_cap:
                capped_chain_ids.add(cid)
        for cid, do_cap in cap_c.items():
            if do_cap:
                capped_chain_ids.add(cid)
        if capped_chain_ids:
            self.missingTerminals = {
                res: atoms for res, atoms in self.missingTerminals.items()
                if res.chain.id not in capped_chain_ids
            }

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
            self.msg(f'Add  missing residues:\n {self.missing_residues_added}')
        if self.missing_residues_skipped:
            self.msg(f'Skip missing residues:\n {self.missing_residues_skipped}')
        if self.missing_atoms_added:
            self.msg(f'Add  missing atoms:\n {self.missing_atoms_added}')

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
                if isinstance(res_id, str) and res_id[-1].isalpha():
                    icode = res_id[-1]
                    res_id = res_id[:-1]
            seq_num = int(res_id) if isinstance(res_id, str) else res_id
            icode_char = (icode or ' ')[:1]
            modres_lines.append(
                f"MODRES {pdb_id:>4} {res_name:3} {chain:1} {seq_num:>4}{icode_char:1} {std_res_name:3}   MODIFIED RESIDUE"
            )
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
        system = forcefield.createSystem(self.topology, nonbondedMethod=app.CutoffNonPeriodic, constraints=None, rigidWater=False)

        current_positions_angstrom = np.array(self.positions.value_in_unit(unit.angstroms))
        for atom in self.topology.atoms():
            atom_pos = current_positions_angstrom[atom.index]
            dists = np.linalg.norm(self._initial_positions_angstrom - atom_pos, axis=1)
            if np.min(dists) < 0.001:
                system.setParticleMass(atom.index, 0.0)
                continue
            logger.info(f"Find added atom: {atom}")

        # # Original logic (commented out):
        # missing_atoms_info_as_dict = defaultdict(list)
        # for chain, res_id, res_name, atoms in self.missing_atoms_added:
        #     missing_atoms_info_as_dict[(chain, res_id, res_name)] += atoms
        # nonstd_names = [res.name for res, stdname in self.nonstandardResidues]
        # for residue in self.topology.residues():
        #     resdata = (residue.chain.id, int(residue.id), residue.name)
        #     if resdata in self.missing_residues_added:
        #         logger.info('Found fixed residue: %s', residue)
        #         continue
        #
        #     for i, atom in enumerate(residue.atoms()):
        #         if residue.name in nonstd_names:
        #             system.setParticleMass(atom.index, 0.0)
        #             continue
        #         if (resdata in missing_atoms_info_as_dict) and (atom.name in missing_atoms_info_as_dict[resdata]):
        #             logger.info('Found fixed atom: %s', atom)
        #             continue
        #         if atom.element is app.element.hydrogen:
        #             continue
        #         # if (self._has_ligand) and (residue.index == self.topology.getNumResidues() - 1) and (i in self.ligand_missing_atoms):
        #         #     continue
        #         system.setParticleMass(atom.index, 0.0)

        integrator = mm.LangevinIntegrator(300*unit.kelvin, 10/unit.picosecond, 5*unit.femtosecond)
        context = mm.Context(system, integrator)
        context.setPositions(self.positions)
        mm.LocalEnergyMinimizer.minimize(context, tolerance=10)
        self.positions = context.getState(getPositions=True).getPositions()
        return self.positions

    def selectResiduesToKeep(
        self,
        keep_chains: Optional[List[str]] = None,
        keep_water: bool = True,
        keep_ions: bool = True,
        extra_keep: Optional[List[str]] = None,
        water_ion_distance: Optional[float] = None,
    ):
        """
        Select which components to keep and remove the rest.

        Handles polymer chains, heterogens, water, and ions. In wizard mode,
        prompts the user interactively for each decision. In non-wizard mode,
        uses the provided parameters.

        Parameters
        ----------
        keep_chains : List[str], optional
            Chain IDs of polymer chains to keep. If ``None``, keep all
            polymer chains (default: ``None``). Only used in non-wizard mode.
        keep_water : bool, optional
            If ``True``, retain water residues (default: ``True``).
            Only used in non-wizard mode.
        keep_ions : bool, optional
            If ``True``, retain ion residues (default: ``True``).
            Only used in non-wizard mode.
        extra_keep : List[str], optional
            Additional heterogen residue names to retain (e.g., ``["OCT"]``).
            Only used in non-wizard mode (default: ``None``).
        water_ion_distance : float, optional
            If set, only keep water and ions within this distance (Angstrom)
            from kept residues. Overrides ``keep_water`` and ``keep_ions``
            (default: ``None``). Only used in non-wizard mode.

        Notes
        -----
        Modifies :attr:`self.topology` and :attr:`self.positions` in-place.
        """
        extra_keep = list() if extra_keep is None else extra_keep
        components = annotate_topology(self.topology)

        self.msg("")
        self.msg("  Structure summary")
        self.msg("  " + "=" * 52)
        for i, (content, comp, chain_id, nres) in enumerate(components):
            self.msg(f"  [{i}]  {content:<10}  chain {chain_id}  ({nres} residues)")
        self.msg("  " + "=" * 52)
        self.msg("")

        keep_residues = []
        polymers = [data for data in components if data[0] in ('Protein', 'DNA', 'RNA')]
        heterogens = [data for data in components if data[0] not in ('Protein', 'DNA', 'RNA', 'Water', 'Ion')]
        water_ions = [data for data in components if data[0] in ('Water', 'Ion')]

        if self.wizard:
            # --- Polymer selection ---
            if polymers:
                self.msg("  Polymers (protein / DNA / RNA)")
            for data in polymers:
                while True:
                    ans = input(
                        f"  Keep {data[0]} (chain {data[2]}, {data[3]} residues)? [Y/n]: "
                    ).strip()
                    if ans.lower() == 'n':
                        break
                    if ans.lower() == 'y' or ans == '':
                        keep_residues.extend(data[1])
                        break
                    print("  -> Enter y or n")

            # --- Heterogen selection ---
            if heterogens:
                self.msg("")
                self.msg("  Heterogens")
            for data in heterogens:
                while True:
                    ans = input(
                        f"  Keep {data[0]} (chain {data[2]})? [y/N]: "
                    ).strip()
                    if ans.lower() == 'y':
                        keep_residues.extend(data[1])
                        break
                    if ans.lower() == 'n' or ans == '':
                        break
                    print("  -> Enter y or n")

            # --- Water & ion selection ---
            if water_ions:
                n_water = sum(d[3] for d in water_ions if d[0] == 'Water')
                n_ion = sum(d[3] for d in water_ions if d[0] == 'Ion')
                self.msg("")
                self.msg("  Water & ions" + (f"  ({n_water} water, {n_ion} ions)" if (n_water or n_ion) else ""))
                while True:
                    option = input(
                        "    1 Keep all    2 Remove all    3 Keep within distance    4 Choose by chain\n"
                        "  Choice [1-4]: "
                    ).strip()
                    if option == '1':
                        for data in water_ions:
                            keep_residues.extend(data[1])
                        break
                    if option == '2':
                        break
                    if option == '3':
                        while True:
                            dist_in = input("  Distance cutoff (Angstrom): ").strip()
                            try:
                                dist = float(dist_in)
                                if dist > 0:
                                    break
                            except ValueError:
                                pass
                            print("  -> Enter a positive number")
                        all_wi = [res for data in water_ions for res in data[1]]
                        keep_residues.extend(self._filter_by_distance(keep_residues, all_wi, dist))
                        break
                    if option == '4':
                        for data in water_ions:
                            while True:
                                ans = input(
                                    f"  Keep {data[0]} (chain {data[2]}, {data[3]} residues)? [Y/n]: "
                                ).strip()
                                if ans.lower() == 'n':
                                    break
                                if ans.lower() == 'y' or ans == '':
                                    keep_residues.extend(data[1])
                                    break
                                print("  -> Enter y or n")
                        break
                    print("  -> Enter 1, 2, 3, or 4")
        else:
            # --- Non-wizard: parameter-driven selection ---
            for data in polymers:
                if keep_chains is None or data[2] in keep_chains:
                    keep_residues.extend(data[1])

            for data in heterogens:
                if data[0] in extra_keep:
                    keep_residues.extend(data[1])

            if water_ion_distance is not None:
                all_wi = [res for data in water_ions for res in data[1]]
                keep_residues.extend(self._filter_by_distance(keep_residues, all_wi, water_ion_distance))
            else:
                for data in water_ions:
                    if (data[0] == 'Water' and keep_water) or (data[0] == 'Ion' and keep_ions):
                        keep_residues.extend(data[1])

        keep_set = set(keep_residues)
        toDelete = [r for r in self.topology.residues() if r not in keep_set]
        if toDelete:
            modeller = app.Modeller(self.topology, self.positions)
            modeller.delete(toDelete)
            self.topology = modeller.topology
            self.positions = modeller.positions

    def run(
        self,
        # selectResiduesToKeep params
        keep_chains: Optional[List[str]] = None,
        keep_water: bool = True,
        keep_ions: bool = True,
        extra_keep: Optional[List[str]] = None,
        water_ion_distance: Optional[float] = None,
        # fixNonStandardResidues params
        replace_nonstandard: bool = True,
        # findMissingAtoms params
        skip_missing_terminal_residues: bool = False,
        max_num_consecutive_missing_residues: Optional[int] = None,
        cap_gaps: bool = True,
        force_cap_terminals: bool = False,
        # hydrogen / refinement
        pH: float = 7.4,
        forcefield=None,
        # output
        res_num_mapping: Optional[Dict[str, Dict[int, str]]] = None,
        out: str | Path | TextIO | None = None
    ) -> str | None:
        """
        Run the complete protein fixing pipeline.

        Executes the full workflow to fix a PDB structure: selecting
        components to keep, replacing non-standard residues, finding and
        adding missing atoms/residues, adding hydrogens, refining positions,
        and writing the output file with appropriate REMARK records.

        In wizard mode, interactively prompts the user for each decision.
        In non-wizard mode, uses the provided parameters.

        Parameters
        ----------
        keep_chains : List[str], optional
            Chain IDs of polymer chains to keep. If ``None``, keep all
            polymer chains (default: ``None``).
            See :meth:`selectResiduesToKeep`.
        keep_water : bool, optional
            If ``True``, retain water residues (default: ``True``).
            See :meth:`selectResiduesToKeep`.
        keep_ions : bool, optional
            If ``True``, retain ion residues (default: ``True``).
            See :meth:`selectResiduesToKeep`.
        extra_keep : List[str], optional
            Additional heterogen residue names to retain (default: ``None``).
            See :meth:`selectResiduesToKeep`.
        water_ion_distance : float, optional
            If set, only keep water/ions within this distance (Angstrom) from
            kept residues (default: ``None``).
            See :meth:`selectResiduesToKeep`.
        replace_nonstandard : bool, optional
            If ``True``, replace all non-standard residues with standard
            equivalents (default: ``True``).
            See :meth:`fixNonStandardResidues`.
        skip_missing_terminal_residues : bool, optional
            If ``True``, skip adding missing residues at chain termini
            (default: ``False``). See :meth:`findMissingAtoms`.
        max_num_consecutive_missing_residues : int, optional
            Maximum number of consecutive missing residues to add
            (default: ``None``). See :meth:`findMissingAtoms`.
        cap_gaps : bool, optional
            If ``True``, add ``ACE``/``NME`` caps to skipped gaps
            (default: ``True``). See :meth:`findMissingAtoms`.
        force_cap_terminals : bool, optional
            If ``True``, force ``ACE``/``NME`` caps on all chain termini
            (default: ``False``). See :meth:`findMissingAtoms`.
        pH : float, optional
            pH for protonation states (default: ``7.4``).
            See :meth:`addMissingHydrogens`.
        forcefield : ForceField, optional
            OpenMM ForceField object (default: ``None``).
        res_num_mapping : Dict[str, Dict[int, str]], optional
            Mapping to restore original PDB residue numbering in output
            (default: ``None``).
        out : str or Path or TextIO or None, optional
            Output destination. File path, file-like object, or ``None``
            to return the PDB string (default: ``None``).

        Returns
        -------
        str or None
            PDB content as string if ``out`` is ``None``, otherwise ``None``.
        """
        self.selectResiduesToKeep(
            keep_chains=keep_chains,
            keep_water=keep_water,
            keep_ions=keep_ions,
            extra_keep=extra_keep,
            water_ion_distance=water_ion_distance,
        )
        self.fixNonStandardResidues(replace_nonstandard=replace_nonstandard)
        self.findMissingAtoms(
            skip_missing_terminal_residues=skip_missing_terminal_residues,
            max_num_consecutive_missing_residues=max_num_consecutive_missing_residues,
            cap_gaps=cap_gaps,
            force_cap_terminals=force_cap_terminals,
        )
        self._initial_positions_angstrom = np.array(self.positions.value_in_unit(unit.angstroms))
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