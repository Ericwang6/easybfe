"""
Small molecule force field parameterization package.

This package provides tools for parameterizing small molecules with various
force fields including GAFF, GAFF2, and OpenFF. It supports automatic charge
assignment and generates topology files in AMBER and OpenMM XML formats.
"""
import os
import logging
import warnings
from typing import Any, Union
from pathlib import Path

from .registry import PARAMETRIZER_REGISTRY
from .base import SmallMoleculeForceField
from .custom import CustomForceField
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


def load_parametrizer(forcefield: str, charge_method: str, engine: str = '', **kwargs) -> SmallMoleculeForceField:
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
        return PARAMETRIZER_REGISTRY.create(engine, forcefield, charge_method, **kwargs)
    except KeyError:
        available = ", ".join(PARAMETRIZER_REGISTRY.names())
        msg = (
            f"Engine {engine!r} is not available. Available: {available}."
            + (" If you requested 'openff', ensure openff-toolkit and openmmforcefields are installed." if engine == "openff" else "")
        )
        raise NotImplementedError(msg)


def parametrize_ligand_single(
    ligand: Union[str, Path, Any],
    name: str,
    out_dir: Union[str, Path],
    forcefield: str,
    charge_method: str,
    engine: str = '',
    **kwargs
) -> Ligand:
    """
    Load, parametrize, and save a ligand with the specified force field.
    
    This function provides a convenient high-level interface for the complete
    ligand parametrization workflow. It loads a ligand from various input formats,
    assigns it a name, generates force field parameters, and saves all output
    files to the specified directory.
    
    Parameters
    ----------
    ligand : str, Path, or Any
        Ligand input source. Supported types:
        
        * :class:`str` or :class:`pathlib.Path`: File path (SDF, CSV, SMILES)
        * :class:`pandas.DataFrame`: DataFrame with name and SMILES columns
        * :class:`list` of :class:`rdkit.Chem.Mol`: List of RDKit molecules
        
        See :class:`easybfe.core.ligand.LigandLoader` for full supported formats.
    name : str
        Name to assign to the ligand. This will override any existing name
        from the input source.
    out_dir : str or Path
        Output directory path where parametrized files will be saved.
        Directory will be created if it does not exist.
    forcefield : str
        Force field name or path. Auto-detection rules:
        
        * Contains 'gaff': :class:`easybfe.smff.gaff.GAFF` (engine ``acpype``)
        * Contains 'openff' or ends with '.xml': :class:`easybfe.smff.openff.OpenFF`
        * Otherwise: :class:`easybfe.smff.custom.CustomForceField`
    charge_method : str
        Partial charge assignment method (e.g., 'bcc', 'gas', 'am1bcc').
    engine : str, optional
        Explicit engine override ('acpype', 'openff', or 'custom'). If empty,
        engine is auto-detected from `forcefield`.
    **kwargs
        Additional keyword arguments passed to the parameterizer constructor
        or :meth:`easybfe.core.ligand.LigandLoader.load`.
    
    Returns
    -------
    Ligand
        Parametrized ligand object with topology files (prmtop, inpcrd, pdb, xml)
        stored as auxiliary files.
    
    Raises
    ------
    ValueError
        If ligand cannot be loaded, has invalid format, or duplicate names are
        found when loading multiple molecules.
    NotImplementedError
        If the specified or auto-detected engine is not supported.
    AssertionError
        If parametrization validation fails (energy mismatch between prmtop and XML).
    
    Examples
    --------
    >>> from easybfe.smff import parametrize_ligand
    >>> # Parametrize from SDF file
    >>> ligand = parametrize_ligand(
    ...     ligand='ligand.sdf',
    ...     name='benzene',
    ...     out_dir='./output',
    ...     forcefield='gaff2',
    ...     charge_method='bcc'
    ... )
    >>> # Parametrize with explicit engine
    >>> ligand = parametrize_ligand(
    ...     ligand='molecule.sdf',
    ...     name='molecule',
    ...     out_dir='./output',
    ...     forcefield='openff-2.1.0',
    ...     charge_method='gas',
    ...     engine='openff'
    ... )
    
    See Also
    --------
    :func:`easybfe.smff.load_parametrizer` : Load a force field parameterizer.
    :class:`easybfe.core.ligand.LigandLoader` : Load ligands from various sources.
    :class:`easybfe.smff.base.SmallMoleculeForceField` : Base parameterizer interface.
    """
    loader = LigandLoader()
    ligands = loader.load(ligand, only_first=True, **kwargs)
    if not ligands:
        raise ValueError(f"No ligands loaded from source: {ligand}")
    if len(ligands) > 0:
        warnings.warn('Multiple ligands found, only the first one will be parameterized.')
    
    ligand_obj = ligands[0]
    ligand_obj.name = name
    
    parametrizer = load_parametrizer(forcefield, charge_method, engine, **kwargs)
    parametrized_ligand = parametrizer.run(ligand_obj)
    
    parametrized_ligand.dump(out_dir)
    
    return parametrized_ligand