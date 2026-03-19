"""
Small molecule force field parameterization package.

This package provides tools for parameterizing small molecules with various
force fields including GAFF, GAFF2, OpenFF, and custom XML-based force fields.
It supports automatic charge assignment, parallel parametrization of multiple
ligands, and generation of topology files in AMBER (prmtop/inpcrd) and OpenMM
XML formats.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any

from .base import SmallMoleculeForceField
from .custom import CustomForceField
from .registry import PARAMETRIZER_REGISTRY
try:
    from .gaff import GAFF
except ImportError:
    pass
try:
    from .openff import OpenFF
except ImportError:
    pass
from ..core.ligand import Ligand, LigandLoader


logger = logging.getLogger(__name__)


def load_parametrizer(forcefield: str, charge_method: str, engine: str = '', raise_errors: bool = False, **kwargs) -> SmallMoleculeForceField:
    """
    Load a force field parameterizer based on force field name and engine.

    When ``engine`` is empty, it is auto-detected from ``forcefield``:

    1. If ``forcefield`` is exactly ``'gaff'`` or ``'gaff2'``, engine is ``'acpype'``
       (:class:`easybfe.smff.gaff.GAFF`).
    2. Else if ``'openff'`` is in ``forcefield``, or ``forcefield`` is an existing file path,
       or ``forcefield`` ends with ``'.xml'``, engine is ``'openff'``
       (:class:`easybfe.smff.openff.OpenFF`).
    3. Otherwise engine is ``'custom'`` (:class:`easybfe.smff.custom.CustomForceField`).

    Parameters
    ----------
    forcefield : str
        Force field name or path (e.g. ``'gaff2'``, ``'openff-2.1.0'``, or path to a topology file).
    charge_method : str
        Partial charge assignment method (e.g. ``'bcc'``, ``'gas'``).
    engine : str, optional
        Explicit engine override: ``'acpype'``, ``'openff'``, or ``'custom'``. If empty,
        engine is auto-detected from ``forcefield`` as above.
    **kwargs
        Additional keyword arguments passed to the parameterizer constructor.

    Returns
    -------
    SmallMoleculeForceField
        Instance of the appropriate parameterizer class.

    Raises
    ------
    NotImplementedError
        If the requested engine is not available (e.g. not installed or not in PATH).
        Message includes available engines and a hint for ``'openff'`` when missing.

    Examples
    --------
    >>> # Auto-detect acpype (GAFF) from forcefield name
    >>> ff = load_parametrizer('gaff2', 'bcc')
    >>> # Auto-detect openff from name or file path
    >>> ff = load_parametrizer('openff-2.1.0', 'bcc')
    >>> ff = load_parametrizer('/path/to/file.xml', 'gas')
    >>> # Explicit engine
    >>> ff = load_parametrizer('openff-2.1.0', 'bcc', engine='openff')
    >>> ff = load_parametrizer('/path/to/topology.prmtop', '', engine='custom')

    See Also
    --------
    :class:`easybfe.smff.gaff.GAFF` : GAFF-based parameterizer (engine ``acpype``).
    :class:`easybfe.smff.openff.OpenFF` : OpenFF-based parameterizer.
    :class:`easybfe.smff.custom.CustomForceField` : Custom force field parameterizer.
    :class:`easybfe.core.ligand.LigandLoader` : Load ligands from various sources.
    """
    if not engine:
        if forcefield == 'gaff' or forcefield == 'gaff2':
            engine = 'acpype'
        elif ('openff' in forcefield) or os.path.isfile(forcefield) or forcefield.endswith('.xml'):
            engine = 'openff'
        else:
            engine = 'custom'

    logger.debug(f"Loading parameterizer: engine={engine}, forcefield={forcefield}, charge_method={charge_method}")
    try:
        return PARAMETRIZER_REGISTRY.create(engine, forcefield, charge_method, raise_errors, **kwargs)
    except KeyError:
        available = ", ".join(PARAMETRIZER_REGISTRY.names())
        msg = (
            f"Engine {engine!r} is not available. Available: {available}."
            + (" If you requested 'openff', ensure openff-toolkit and openmmforcefields are installed." if engine == "openff" else "")
        )
        raise NotImplementedError(msg)


def parametrize_ligands(
    source: Any,
    output: str | Path | None = None,
    output_base: str | Path | None = None,
    forcefield: str = "gaff2",
    charge_method: str = "bcc",
    engine: str = "",
    raise_errors: bool = False,
    nprocs: int = -1,
    resp_engine: str = "",
    keep_cache: bool = False,
    **kwargs: Any,
) -> list[Ligand]:
    """
    Load, parametrize, and save one or more ligands with the specified force field.

    This high-level convenience function wraps :class:`SmallMoleculeForceField`
    implementations and :class:`easybfe.core.ligand.LigandLoader` to perform the
    complete parametrization workflow:

    1. Load ligands from a flexible ``source`` (files, SMILES, RDKit molecules, etc.).
    2. Construct an appropriate parameterizer via :func:`load_parametrizer`.
    3. Route to single- or multi-ligand parametrization based on loaded count and
       output arguments.

    Parameters
    ----------
    source : Any
        Ligand input source passed directly to
        :meth:`easybfe.core.ligand.LigandLoader.load`. Typical values include:

        * :class:`str` or :class:`pathlib.Path`: File path (SDF, CSV, SMILES, etc.)
        * Sequence of :class:`rdkit.Chem.Mol`
        * Other objects supported by :class:`LigandLoader`.
    output : str, Path, or None, optional
        Direct output directory for a single ligand. Files are written directly
        under this path. Only valid when exactly one ligand is loaded. If both
        ``output`` and ``output_base`` are provided, ``output`` takes precedence
        and a warning is emitted.
    output_base : str, Path, or None, optional
        Base directory for per-ligand output subdirectories. Each ligand is
        written under ``output_base / ligand.name``. Required when multiple
        ligands are loaded or when ``output`` is not provided.
    forcefield : str, default='gaff2'
        Force field name or path. When ``engine`` is empty, the engine is
        auto-detected from ``forcefield`` (see :func:`load_parametrizer`).
    charge_method : str, default='bcc'
        Partial charge assignment method (e.g. ``'bcc'``, ``'gas'``, ``'resp'``).
    engine : str, optional
        Explicit engine override (``'acpype'``, ``'openff'``, or ``'custom'``).
        If empty, the engine is auto-detected from ``forcefield``.
    raise_errors : bool, default=False
        If ``True``, parametrization errors are raised immediately. If ``False``,
        errors are logged/warned and failed ligands may be omitted from the
        returned list.
    nprocs : int, default=-1
        Number of parallel processes. If -1, uses all available CPUs. If 1, runs
        sequentially.
    resp_engine : str, optional
        Engine for RESP charge calculations (e.g. ``'qchem'``). Only used when
        ``charge_method`` starts with ``'resp'``.
    keep_cache : bool, default=False
        If ``True``, keep the intermediate ``.smff.tmp`` working directory after
        parametrization.
    **kwargs
        Additional keyword arguments forwarded to
        :meth:`easybfe.core.ligand.LigandLoader.load` (e.g. ``only_first=True``,
        ``name_from_stem=True``).

    Returns
    -------
    list[Ligand]
        List of successfully parametrized ligands.

    Raises
    ------
    ValueError
        If no ligands can be loaded from ``source``, or if ``output_base`` is
        required but not provided.
    NotImplementedError
        If the specified or auto-detected engine is not supported.

    See Also
    --------
    :func:`easybfe.smff.load_parametrizer`
        Load a force field parameterizer implementation.
    :class:`easybfe.core.ligand.LigandLoader`
        Flexible ligand loading from files/SMILES/RDKit molecules.
    :class:`easybfe.smff.base.SmallMoleculeForceField`
        Base parameterizer interface.
    """
    loader = LigandLoader()
    ligands = loader.load(source, **kwargs)
    if not ligands:
        raise ValueError(f"No ligands loaded from source: {source!r}")

    logger.info("Loaded %d ligand(s) from source %r", len(ligands), source)

    parametrizer = load_parametrizer(
        forcefield, charge_method, engine, raise_errors,
        resp_engine=resp_engine, keep_cache=keep_cache,
    )

    is_single = len(ligands) == 1

    if is_single and output is not None:
        if output_base is not None:
            warnings.warn(
                "--output (-o) will override --output-base (-O) for single ligand mode.",
                UserWarning,
            )
        result = parametrizer.run(ligands[0], output, nprocs)
        return [result] if result is not None else []
    else:
        if output_base is None:
            raise ValueError(
                "--output-base (-O) is required when parametrizing multiple ligands "
                "or when --output (-o) is not provided."
            )
        return parametrizer.run(ligands, output_base, nprocs)