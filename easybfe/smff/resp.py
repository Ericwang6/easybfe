"""
RESP charge calculation via Q-Chem.

This module provides utilities to compute Merz-Kollman RESP charges using
Q-Chem and apply them to an existing AMBER topology.  The workflow is:

1. Generate a Q-Chem single-point input with ``RESP_CHARGES = 1``.
2. Execute Q-Chem via :func:`run_qchem`.
3. Parse the resulting charges from the output file.
4. Replace charges in a ParmEd structure and re-save.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import parmed
from rdkit import Chem

from ..cmd import find_executable, run_command

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_FUNCTIONAL = "HF"
_DEFAULT_BASIS = "6-31G*"


def parse_resp_spec(charge_method: str) -> tuple[str, str]:
    """Parse a ``resp``-prefixed charge method into functional and basis set.

    Parameters
    ----------
    charge_method : str
        Either ``"resp"`` (uses HF/6-31G*) or ``"resp/<functional>/<basis>"``.

    Returns
    -------
    functional : str
    basis : str

    Raises
    ------
    ValueError
        If the string does not start with ``"resp"`` or has wrong number of
        ``/``-separated components.
    """
    parts = charge_method.split("/")
    if parts[0] != "resp":
        raise ValueError(
            f"charge_method must start with 'resp', got {charge_method!r}"
        )
    if len(parts) == 1:
        return _DEFAULT_FUNCTIONAL, _DEFAULT_BASIS
    if len(parts) == 3:
        return parts[1], parts[2]
    raise ValueError(
        f"Expected 'resp' or 'resp/<functional>/<basis>', got {charge_method!r}"
    )


def generate_qchem_input(
    mol: Chem.Mol,
    functional: str = _DEFAULT_FUNCTIONAL,
    basis: str = _DEFAULT_BASIS,
    net_charge: int = 0,
    multiplicity: int = 1,
) -> str:
    """Build a Q-Chem input string for a RESP charge calculation.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule with at least one 3-D conformer and explicit hydrogens.
    functional : str
        Electronic-structure method (e.g. ``"HF"``, ``"B3LYP"``).
    basis : str
        Basis set (e.g. ``"6-31G*"``).
    net_charge : int
        Net molecular charge.
    multiplicity : int
        Spin multiplicity.

    Returns
    -------
    str
        Complete Q-Chem input file content.
    """
    conf = mol.GetConformer()
    lines = [
        "$molecule",
        f"{net_charge} {multiplicity}",
    ]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"  {atom.GetSymbol():2s}  {pos.x:14.8f}  {pos.y:14.8f}  {pos.z:14.8f}")
    lines.append("$end")
    lines.append("")
    lines.append("$rem")
    lines.append("JOBTYPE              SP")
    lines.append(f"METHOD               {functional}")
    lines.append(f"BASIS                {basis}")
    lines.append("RESP_CHARGES         1")
    lines.append("SYMMETRY_DECOMPOSITION 0")
    lines.append("MEM_TOTAL  64000")
    lines.append("MEM_STATIC 10000")
    lines.append("SCF_CONVERGENCE  8")
    lines.append("THRESH  14")
    lines.append("$end")
    lines.append("")
    return "\n".join(lines)


def run_qchem(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    nproc: int = 1,
) -> None:
    """Execute Q-Chem on *input_path*, writing results to *output_path*.

    Parameters
    ----------
    input_path : path-like
        Q-Chem ``.in`` file.
    output_path : path-like
        Destination for Q-Chem output.
    nproc : int
        Number of OpenMP threads (``qchem -nt``).

    Raises
    ------
    CommandExecuteError
        If Q-Chem returns a non-zero exit code.
    """
    qchem = find_executable("qchem")
    cmd = [qchem, "-nt", str(nproc), str(input_path), str(output_path)]
    logger.info("Running Q-Chem: %s", " ".join(cmd))
    run_command(cmd, raise_error=True)


_RESP_HEADER = "Merz-Kollman RESP Net Atomic Charges"
_CHARGE_RE = re.compile(r"\s+\d+\s+\w+\s+(-?\d+\.\d+)")
_SEPARATOR = re.compile(r"^\s*-{5,}\s*$")


def parse_resp_charges(output_path: str | os.PathLike) -> list[float]:
    """Extract RESP atomic charges from a Q-Chem output file.

    Parameters
    ----------
    output_path : path-like
        Path to the Q-Chem output file.

    Returns
    -------
    list of float
        Per-atom RESP charges in atom order.

    Raises
    ------
    RuntimeError
        If the RESP charge section cannot be found in the output.
    """
    text = Path(output_path).read_text()
    idx = text.find(_RESP_HEADER)
    if idx == -1:
        raise RuntimeError(
            f"Could not find '{_RESP_HEADER}' section in {output_path}"
        )

    charges: list[float] = []
    in_table = False
    for line in text[idx:].splitlines():
        if _SEPARATOR.match(line):
            if in_table:
                break
            in_table = True
            continue
        if in_table:
            m = _CHARGE_RE.match(line)
            if m:
                charges.append(float(m.group(1)))

    if not charges:
        raise RuntimeError(
            f"Parsed zero charges from '{_RESP_HEADER}' section in {output_path}"
        )
    return charges


def apply_resp_charges(
    prmtop_path: str | os.PathLike,
    inpcrd_path: str | os.PathLike,
    charges: list[float],
) -> tuple[str, str]:
    """Replace atomic charges in an AMBER topology with RESP values.

    Parameters
    ----------
    prmtop_path : path-like
        Path to the AMBER prmtop file (will be overwritten).
    inpcrd_path : path-like
        Path to the AMBER inpcrd file (re-saved unchanged).
    charges : list of float
        Per-atom RESP charges (elementary-charge units, same order as topology).

    Returns
    -------
    prmtop_text : str
        Updated prmtop file content.
    inpcrd_text : str
        inpcrd file content (unchanged).

    Raises
    ------
    ValueError
        If ``len(charges)`` does not match the number of atoms in the topology.
    """
    struct = parmed.load_file(str(prmtop_path), xyz=str(inpcrd_path))
    if len(charges) != len(struct.atoms):
        raise ValueError(
            f"Number of RESP charges ({len(charges)}) does not match "
            f"number of atoms in topology ({len(struct.atoms)})"
        )
    for atom, q in zip(struct.atoms, charges):
        atom.charge = q

    struct.save(str(prmtop_path), overwrite=True)
    struct.save(str(inpcrd_path), overwrite=True)

    prmtop_text = Path(prmtop_path).read_text()
    inpcrd_text = Path(inpcrd_path).read_text()
    return prmtop_text, inpcrd_text
