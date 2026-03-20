import openmm.app as app
from pathlib import Path
from textwrap import wrap

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
