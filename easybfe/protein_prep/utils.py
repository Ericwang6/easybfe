import openmm.app as app
from pathlib import Path
from textwrap import wrap
from pdbfixer import PDBFixer

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


def convert_to_seqres(sequence: list[str], chain_id: str):
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


def annotate_topology(top: app.Topology):
    components = []
    for i, chain in enumerate(top.chains()):
        res_list = list(chain.residues())
        residues = [r.name for r in res_list]
        if any(r in proteinResidues for r in residues):
            content = "Protein"
            components.append((content, res_list, chain.id, len(res_list)))
        elif any(r in rnaResidues for r in residues):
            content = "RNA"
            components.append((content, res_list, chain.id, len(res_list)))
        elif any(r in dnaResidues for r in residues):
            content = "DNA"
            components.append((content, res_list, chain.id, len(res_list)))
        else:
            waters = []
            ions = []
            for residue in chain.residues():
                if residue.name in waterResidues:
                    waters.append(residue)
                elif residue.name in ionResidues:
                    ions.append(residue)
                else:
                    components.append((residue.name, [residue], chain.id, 1))
            if waters:
                components.append(('Water', waters, chain.id, len(waters)))
            if ions:
                components.append(('Ion', ions, chain.id, len(ions)))
    return components


def check_ff(
    protein_file: str | Path,
    forcefield_files: str | list[str] | tuple[str, ...] | None = None,
) -> bool:
    """
    Validate that a force field can parameterize a protein topology.

    Parameters
    ----------
    protein_file : str or Path
        Input protein structure file in PDB format.
    forcefield_files : str or list[str] or tuple[str, ...], optional
        OpenMM force field XML file(s). If ``None``, defaults to
        ``("amber14-all.xml", "amber14/tip3p.xml")``.

    Returns
    -------
    bool
        ``True`` if system creation succeeds.

    Raises
    ------
    Exception
        Raised by OpenMM when the topology cannot be parameterized by
        the given force field combination.
    """
    if forcefield_files is None:
        forcefield_files = ("amber14-all.xml", "amber14/tip3p.xml")
    elif isinstance(forcefield_files, str):
        forcefield_files = (forcefield_files,)
    else:
        forcefield_files = tuple(forcefield_files)

    pdb = app.PDBFile(str(protein_file))
    ff = app.ForceField(*forcefield_files)
    ff.createSystem(pdb.topology)
    return True


def summary_pdb(protein_file: str | Path) -> str:
    """
    Summarize a PDB structure without modifying coordinates.

    The summary includes:

    - Chain/component overview from :func:`annotate_topology`
    - Non-standard residues detected by :class:`pdbfixer.PDBFixer`
    - Missing residues detected from SEQRES vs ATOM records
    - Missing atoms and terminal atoms

    Parameters
    ----------
    protein_file : str or Path
        Input protein structure file in PDB format.

    Returns
    -------
    str
        Human-readable multi-line summary text.
    """
    protein_file = Path(protein_file)
    fixer = PDBFixer(filename=str(protein_file))
    lines: list[str] = []

    lines.append(f"Structure summary for: {protein_file}")
    lines.append("=" * 64)

    components = annotate_topology(fixer.topology)
    lines.append("Components")
    lines.append("-" * 64)
    for i, (content, _residues, chain_id, nres) in enumerate(components):
        lines.append(f"[{i}] {content:<10} chain {chain_id} ({nres} residues)")

    fixer.findNonstandardResidues()
    lines.append("")
    lines.append("Non-standard residues")
    lines.append("-" * 64)
    if fixer.nonstandardResidues:
        for residue, std_name in fixer.nonstandardResidues:
            lines.append(
                f"- {_residue_repr(residue)} -> suggested replacement: {std_name}"
            )
    else:
        lines.append("None")

    fixer.findMissingResidues()
    lines.append("")
    lines.append("Missing residues")
    lines.append("-" * 64)
    if fixer.missingResidues:
        chains = list(fixer.topology.chains())
        chain_residues = [list(chain.residues()) for chain in chains]
        for (chain_index, res_id_start), residues in fixer.missingResidues.items():
            chain = chains[chain_index]
            if res_id_start < len(chain_residues[chain_index]):
                gap_start = int(chain_residues[chain_index][res_id_start].id) - len(residues)
            else:
                gap_start = int(chain_residues[chain_index][res_id_start - 1].id) + 1
            gap_end = gap_start + len(residues) - 1
            lines.append(
                f"- Chain {chain.id}: residues {gap_start}-{gap_end} "
                f"({len(residues)} missing) -> {', '.join(residues)}"
            )
    else:
        lines.append("None")

    fixer.findMissingAtoms()
    lines.append("")
    lines.append("Missing atoms")
    lines.append("-" * 64)
    missing_atom_names = {
        residue: [atom.name for atom in atoms]
        for residue, atoms in fixer.missingAtoms.items()
    }
    for residue, atoms in fixer.missingTerminals.items():
        if residue in missing_atom_names:
            missing_atom_names[residue] += atoms
        else:
            missing_atom_names[residue] = list(atoms)

    if missing_atom_names:
        for residue, atom_names in missing_atom_names.items():
            lines.append(f"- {_residue_repr(residue)}: {', '.join(atom_names)}")
    else:
        lines.append("None")

    return "\n".join(lines)
