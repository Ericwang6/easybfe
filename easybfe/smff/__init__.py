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
from .base import SmallMoleculeForceField
from .custom import CustomForceField
from .gaff import GAFF
from .openff import OpenFF
from ..core.ligand import Ligand, LigandLoader


logger = logging.getLogger(__name__)


PARAMETRIZER_REGISTRY = {
    'gaff': GAFF,
    'openff': OpenFF,
    'custom': CustomForceField
}


def load_parametrizer(forcefield: str, charge_method: str, engine: str = '', **kwargs) -> SmallMoleculeForceField:
    """
    Load a force field parameterizer based on force field name and engine.
    
    This function automatically detects the appropriate parameterizer based on
    the force field name, or uses an explicitly specified engine. The selected
    parameterizer will be used to generate force field parameters for small
    molecules.
    
    Parameters
    ----------
    forcefield : str
        Force field name or path. Auto-detection rules:
        
        * Contains 'gaff': :class:`easybfe.smff.gaff.GAFF`
        * Contains 'openff' or ends with '.xml': :class:`easybfe.smff.openff.OpenFF`
        * Otherwise: :class:`easybfe.smff.custom.CustomForceField`
    charge_method : str
        Partial charge assignment method (e.g., 'bcc', 'gas').
    engine : str, optional
        Explicit engine override ('gaff', 'openff', or 'custom'). If empty,
        engine is auto-detected from `forcefield`.
    **kwargs
        Additional keyword arguments passed to the parameterizer constructor.
    
    Returns
    -------
    SmallMoleculeForceField
        Instance of the appropriate parameterizer class.
    
    Raises
    ------
    NotImplementedError
        If the specified or auto-detected engine is not supported.
    
    Examples
    --------
    >>> # Auto-detect GAFF engine
    >>> ff = load_parametrizer('gaff2', 'bcc')
    >>> # Explicitly specify OpenFF engine
    >>> ff = load_parametrizer('openff-2.1.0', 'bcc', engine='openff')
    >>> # Use custom force field from XML file
    >>> ff = load_parametrizer('custom.xml', '', engine='custom')
    
    See Also
    --------
    :class:`easybfe.smff.gaff.GAFF` : GAFF-based parameterizer.
    :class:`easybfe.smff.openff.OpenFF` : OpenFF-based parameterizer.
    :class:`easybfe.smff.custom.CustomForceField` : Custom force field parameterizer.
    :class:`easybfe.ligand.LigandLoader` : Load ligands from files or other sources.
    """
    if not engine:
        if 'gaff' in forcefield:
            engine = 'gaff'
        elif 'openff' in forcefield:
            engine = 'openff'
        elif os.path.isfile(forcefield) and forcefield.endswith('.xml'):
            engine = 'openff'
        else:
            engine = 'custom'
    
    if engine not in PARAMETRIZER_REGISTRY:
        raise NotImplementedError(f"Not supported engine {engine}")

    logger.debug(f"Loading parameterizer: engine={engine}, forcefield={forcefield}, charge_method={charge_method}")
    cls = PARAMETRIZER_REGISTRY[engine]
    return cls(forcefield, charge_method, **kwargs)


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
        
        * Contains 'gaff': :class:`easybfe.smff.gaff.GAFF`
        * Contains 'openff' or ends with '.xml': :class:`easybfe.smff.openff.OpenFF`
        * Otherwise: :class:`easybfe.smff.custom.CustomForceField`
    charge_method : str
        Partial charge assignment method (e.g., 'bcc', 'gas', 'am1bcc').
    engine : str, optional
        Explicit engine override ('gaff', 'openff', or 'custom'). If empty,
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