'''
This package contains small molecule force field parameterizer
'''
import os
import logging
from .base import SmallMoleculeForceField
from .custom import CustomForceField
from .gaff import GAFF
from .openff import OpenFF


logger = logging.getLogger(__name__)


PARAMETRIZER_REGISTRY = {
    'gaff': GAFF,
    'openff': OpenFF,
    'custom': CustomForceField
}


def load_parametrizer(forcefield: str, charge_method: str, engine: str = '', **kwargs) -> SmallMoleculeForceField:
    """
    Load a force field parameterizer based on force field name and engine.
    
    Parameters
    ----------
    forcefield : str
        Force field name or path. If it contains 'gaff', uses GAFF engine.
        If it contains 'openff' or is an XML file, uses OpenFF engine.
        Otherwise, uses custom engine.
    charge_method : str
        Method for assigning partial charges.
    engine : str, optional
        Explicit engine name ('gaff', 'openff', or 'custom'). If not provided,
        will be auto-detected from forcefield.
    **kwargs
        Additional keyword arguments passed to the parameterizer constructor.
    
    Returns
    -------
    SmallMoleculeForceField
        An instance of the appropriate parameterizer class.
    
    Raises
    ------
    NotImplementedError
        If the engine is not supported.
    
    Examples
    --------
    >>> ff = load_parametrizer('gaff2', 'bcc')
    >>> ff = load_parametrizer('openff-2.1.0', 'bcc', engine='openff')
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


def parametrize_ligand(ligand_file: os.PathLike, output_dir: os.PathLike, forcefield: str, charge_method: str, engine: str = ''):
    """
    Parametrize a ligand using the specified force field.
    
    Parameters
    ----------
    ligand_file : os.PathLike
        Path to the ligand structure file.
    output_dir : os.PathLike
        Output directory for parameter files.
    forcefield : str
        Force field name or path.
    charge_method : str
        Method for assigning partial charges.
    engine : str, optional
        Explicit engine name. If not provided, will be auto-detected.
    
    See Also
    --------
    :func:`load_parametrizer` : Function that creates the parameterizer.
    :meth:`easybfe.smff.base.SmallMoleculeForceField.run` : Method that performs parametrization.
    """
    ff = load_parametrizer(forcefield, charge_method, engine)
    ff.run(ligand_file, output_dir)