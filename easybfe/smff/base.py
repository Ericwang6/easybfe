"""
Abstract base classes for small molecule force field parameterization.

This module defines the core interfaces for force field parameterizers and
parametrization jobs. All concrete implementations must inherit from
:class:`SmallMoleculeForceField` and implement the :meth:`_parametrize` method.
"""
import abc
import logging
import os
import shutil
import tempfile
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parmed

from .utils import convert_to_xml
from ..core.ligand import Ligand


logger = logging.getLogger(__name__)


def _openmm_platform_for_validation():
    """Pick a host-only OpenMM platform for prmtop/XML energy checks.

    Default platform (often CUDA/OpenCL) can stall or contend when many
    :class:`multiprocessing.Pool` workers validate in parallel; Reference/CPU
    avoids that for these lightweight single-point energies.
    """
    for name in ("Reference", "CPU"):
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    return None


class SmallMoleculeForceField(abc.ABC):
    """
    Abstract base class for small molecule force field parameterizers.
    
    This class defines the interface for all force field parameterization
    implementations. Subclasses must implement :meth:`_parametrize` to generate
    force field parameters. The base class provides validation and parallel
    execution capabilities.
    
    Parameters
    ----------
    forcefield : str, optional
        Force field identifier or path (interpretation depends on subclass).
    charge_method : str, optional
        Partial charge assignment method (e.g., 'bcc', 'gas', 'am1bcc').
    *args
        Additional positional arguments for subclasses.
    **kwargs
        Additional keyword arguments for subclasses.
    
    Attributes
    ----------
    forcefield : str
        Force field identifier.
    charge_method : str
        Charge assignment method.
    
    Notes
    -----
    Subclasses should implement :meth:`_parametrize` to generate at minimum:
    
    * ``{name}.prmtop``: AMBER topology file
    * ``{name}.inpcrd``: AMBER coordinate file
    
    The base class automatically validates the generated parameters by converting
    to OpenMM XML and comparing energies.
    
    Examples
    --------
    >>> from easybfe.smff import GAFF, OpenFF
    >>> # Use GAFF2 with AM1-BCC charges
    >>> gaff = GAFF('gaff2', 'bcc')
    >>> # Use OpenFF 2.1.0 with Gasteiger charges
    >>> openff = OpenFF('openff-2.1.0', 'gas')
    
    See Also
    --------
    :class:`easybfe.smff.gaff.GAFF` : GAFF/GAFF2 implementation.
    :class:`easybfe.smff.openff.OpenFF` : OpenFF implementation.
    :class:`easybfe.smff.custom.CustomForceField` : Custom force field implementation.
    """
    
    def __init__(self, forcefield: str = '', charge_method: str = '', raise_errors: bool = True, *args, **kwargs):
        """Initialize the force field parameterizer."""
        self.forcefield = forcefield
        self.charge_method = charge_method
        self._raise_errors = raise_errors
    
    def raise_error(self, msg):
        if self._raise_errors:
            raise Exception(msg)
        else:
            warnings.warn(f'Parameterization failed: {msg}')
    
    def _setup(self, ligand: Ligand):
        return ligand.embed(return_new=True)

    def _validate(self, ligand: Ligand, wdir: str):
        """
        Validate parametrization by comparing prmtop and XML energies.
        
        This method ensures the generated force field parameters are consistent
        by converting prmtop/inpcrd to OpenMM XML and comparing potential
        energies. It also generates a PDB file with standardized residue naming.
        
        Parameters
        ----------
        ligand : Ligand
            Ligand object with prmtop and inpcrd files stored as auxiliary files.
        wdir : str
            Working directory path where files will be written for validation.
        
        Returns
        -------
        Ligand
            Ligand object with validated parameters and added pdb/xml auxiliary files.
        
        Raises
        ------
        AssertionError
            If energy difference between prmtop and XML exceeds 0.01 kJ/mol,
            indicating an incompatible force field conversion.
        
        Notes
        -----
        This validation catches issues such as:
        
        * Incorrect unit conversions
        * Unsupported force field terms
        * Corrupted parameter files
        
        The method writes prmtop/inpcrd files to `wdir`, generates pdb/xml files,
        compares energies, and stores pdb/xml as auxiliary files in the ligand object.
        
        See Also
        --------
        :func:`easybfe.smff.utils.convert_to_xml` : XML conversion function.
        """
        logger.info(f"Validating parametrization for {ligand.name}")
        ligand.check_aux_file('prmtop')
        ligand.check_aux_file('inpcrd')
        ligand.dump(wdir)

        prmtop = os.path.join(wdir, f'{ligand.name}.prmtop')
        inpcrd = os.path.join(wdir, f'{ligand.name}.inpcrd')
        pdb = os.path.join(wdir, f'{ligand.name}.pdb')
        xml = os.path.join(wdir, f'{ligand.name}.xml')

        struct = parmed.load_file(prmtop, xyz=inpcrd)
        struct.residues[0].name = 'MOL'
        app.PDBFile.writeFile(struct.topology, struct.positions, pdb, keepIds=True)
        convert_to_xml(struct, xml)

        _plat = _openmm_platform_for_validation()
        if _plat is not None:
            logger.debug(
                "Using OpenMM platform %r for validation of %s",
                _plat.getName(),
                ligand.name,
            )

        system_ref = app.AmberPrmtopFile(prmtop).createSystem()
        integ_ref = mm.LangevinIntegrator(300, 1.0, 0.001)
        ctx_ref = (
            mm.Context(system_ref, integ_ref, _plat)
            if _plat is not None
            else mm.Context(system_ref, integ_ref)
        )
        ctx_ref.setPositions(app.AmberInpcrdFile(inpcrd).positions)
        energy_ref = ctx_ref.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        pdb_obj = app.PDBFile(pdb)
        system = app.ForceField(xml).createSystem(pdb_obj.topology)
        integ_xml = mm.LangevinIntegrator(300, 1.0, 0.001)
        ctx = (
            mm.Context(system, integ_xml, _plat)
            if _plat is not None
            else mm.Context(system, integ_xml)
        )
        ctx.setPositions(app.AmberInpcrdFile(inpcrd).positions)
        energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        energy_diff = abs(energy_ref - energy)
        if energy_diff >= 0.01:
            msg = (
                f"Fail to convert prmtop to xml, the force field might not be compatitable "
                f"because the energy is different {energy_ref} (prmtop) != {energy} (xml)"
            )
            logger.warning(msg)
            warnings.warn(msg)
        logger.info(f"Energy validation passed.")
        logger.debug(f"Energy {energy_ref:.4f} kJ/mol with prmtop and {energy:.4f} with xml")

        ligand.add_aux_file("pdb", Path(pdb).read_text())
        ligand.add_aux_file("xml", Path(xml).read_text())
        ligand.source = f'<{self.__class__.__name__}>'
        return ligand


    def _run(self, ligand: Ligand, wdir: os.PathLike = None):
        """
        Execute the complete parametrization workflow for a single ligand.
        
        This method orchestrates the full parametrization process by calling
        :meth:`_setup`, :meth:`_parametrize` (subclass-specific), and :meth:`_validate`
        (base class validation) in sequence within a temporary directory.
        
        Parameters
        ----------
        ligand : Ligand
            Ligand object to parametrize.
        
        Returns
        -------
        Ligand
            Ligand object with parametrized topology files and validated parameters.
        
        See Also
        --------
        :meth:`run` : Public interface supporting parallel execution.
        """
        tmpd = tempfile.mkdtemp() if wdir is None else Path(wdir).resolve() / '.smff.tmp'
        Path(tmpd).mkdir(exist_ok=True, parents=True)
        pid = os.getpid()
        logger.info(
            "Ligand %s [pid=%s]: stage=setup (tmp=%s)",
            ligand.name,
            pid,
            tmpd,
        )
        # setup
        try:
            ligand = self._setup(ligand)
        except Exception as e:
            self.raise_error(f"Ligand setup failed:\n {traceback.format_exc()}")
            return None

        logger.info(
            "Ligand %s [pid=%s]: stage=force_field (acpype/openff/...)",
            ligand.name,
            pid,
        )
        # parameterize
        try:
            ligand = self._parametrize(ligand, tmpd)
        except Exception as e:
            self.raise_error(f'{traceback.format_exc()}\n\n ERROR: Ligand parameterization failed due to the above reason, please check {tmpd}')
            return None

        logger.info(
            "Ligand %s [pid=%s]: stage=openmm_validate",
            ligand.name,
            pid,
        )
        # validation
        try:
            ligand = self._validate(ligand, tmpd)
        except Exception as e:
            self.raise_error(f'Ligand parameterization failed to be validated because of the following error: {traceback.format_exc()}')
            return None
        
        shutil.rmtree(tmpd)
        if wdir is not None:
            ligand.dump(wdir)
        return ligand
    
    def _run_wrapper(self, args):
        return self._run(*args)
    
    def run(
        self,
        ligand: list[Ligand] | Ligand,
        output_base_dir: str | Path | None = None,
        nprocs: int = -1,
    ) -> Ligand | list[Ligand]:
        """
        Parametrize one or more ligands with optional parallel execution.
        
        Parameters
        ----------
        ligand : Ligand or list of Ligand
            Single ligand or list of ligands to parametrize.
        output_base_dir : str or Path, optional
            Base directory where per-ligand subdirectories will be created. If
            provided, each ligand is written to ``output_base_dir / ligand.name``.
            Duplicate ligand names are automatically disambiguated.
        nprocs : int, default=-1
            Number of parallel processes. If -1, uses all available CPUs.
            If 1, runs sequentially without multiprocessing overhead.
        
        Returns
        -------
        Ligand or list of Ligand
            Parametrized ligand(s) with topology files and validated parameters.
            If the input is a single :class:`Ligand`, a single :class:`Ligand`
            is returned; otherwise a list is returned.
        
        Examples
        --------
        >>> from easybfe.smff import GAFF
        >>> from easybfe.ligand import LigandLoader
        >>> gaff = GAFF('gaff2', 'bcc')
        >>> loader = LigandLoader()
        >>> # Single ligand
        >>> ligands = loader.load('ligand.sdf', only_first=True)
        >>> ligand = gaff.run(ligands[0])
        >>> # Multiple ligands in parallel
        >>> ligands = loader.load(['lig1.sdf', 'lig2.sdf'])
        >>> ligands = gaff.run(ligands, nprocs=4)
        
        See Also
        --------
        :func:`easybfe.parallel.run_func_parallel` : Parallel execution utility.
        :class:`easybfe.ligand.LigandLoader` : Load ligands from files or other sources.
        """
        from ..parallel import run_func_parallel

        # Track whether a single ligand was passed in
        single_input = isinstance(ligand, Ligand)

        # Normalize input to list
        input_ligands = [ligand] if single_input else list(ligand)

        if output_base_dir is not None:
            names_count: dict[str, int] = defaultdict(int)
            input_names: list[str] = []
            output_dirs: list[Path] = []
            base_dir = Path(output_base_dir)
            for i, lig in enumerate(input_ligands):
                if not lig.name:
                    lig.name = f"unnamed_{i}"
                    warnings.warn(
                        f"Ligand {i} has no name, '{lig.name}' is assigned",
                        UserWarning,
                    )
                count = names_count[lig.name]
                if count > 0:
                    new_name = f"{lig.name}_{count}"
                    warnings.warn(
                        f"Duplicated name {lig.name} found. "
                        f"Ligand {i} is renamed to {new_name}",
                        UserWarning,
                    )
                    lig.name = new_name
                names_count[lig.name] += 1
                input_names.append(lig.name)
                output_dirs.append(base_dir / lig.name)
        else:
            output_dirs = [None for _ in range(len(input_ligands))]
            input_names = [lig.name for lig in input_ligands]

        # Run parametrization in parallel (order may not be preserved)
        output_ligands = run_func_parallel(
            self._run_wrapper,
            [(l, p) for l, p in zip(input_ligands, output_dirs)],
            nprocs,
        )

        if not self._raise_errors:
            n_total = len(output_ligands)
            n_failed = sum(lig is None for lig in output_ligands)
            n_success = n_total - n_failed
            logger.info(
                "Parametrization finished: %d succeeded, %d failed.",
                n_success,
                n_failed,
            )

        if len(input_names) != len(set(input_names)):
            warnings.warn(
                "Duplicate ligand names detected in input. Returned ligands are not in "
                "the input order.",
                UserWarning,
            )
            return output_ligands

        # Create mapping from output name to output ligand (skip failures)
        output_name_to_ligand = {
            lig.name: lig for lig in output_ligands if lig is not None
        }

        # Reorder outputs to match input order using name keys
        ordered_outputs: list[Ligand | None] = [
            output_name_to_ligand.get(name) for name in input_names
        ]

        # Filter out failed parametrizations (None)
        ordered_outputs = [lig for lig in ordered_outputs if lig is not None]

        # Return single ligand if input was single, otherwise return list
        if single_input:
            return ordered_outputs[0] if ordered_outputs else None
        return ordered_outputs


    @abc.abstractmethod
    def _parametrize(self, ligand: Ligand, wdir: str) -> Ligand:
        """
        Generate force field parameters for a ligand (subclass implementation).
        
        This abstract method must be implemented by subclasses to perform the
        actual force field parameter generation. Implementations should write
        prmtop and inpcrd files to the working directory and store them as
        auxiliary files in the ligand object.
        
        Parameters
        ----------
        ligand : Ligand
            Ligand object with 3D structure in mol_block. The ligand has already
            been processed by :meth:`_setup` to ensure 3D coordinates are available.
        wdir : str
            Working directory path for writing output files. Files should be written
            using ``os.path.join(wdir, f'{ligand.name}.prmtop')`` format.
        
        Returns
        -------
        Ligand
            Ligand object with prmtop and inpcrd files stored as auxiliary files
            (accessed via ``ligand.auxiliary_files['prmtop']`` and
            ``ligand.auxiliary_files['inpcrd']``).
        
        Notes
        -----
        Implementations have access to ligand attributes:
        
        * ``ligand.name``: Ligand identifier (str)
        * ``ligand.mol_block``: 3D structure in SDF mol block format (str)
        * ``ligand.get_rdmol()``: Get RDKit molecule object with 3D coordinates
        * ``ligand.add_aux_file(filename, content)``: Store file content as string
        
        Required workflow:
        
        1. Extract molecule from ``ligand.get_rdmol()``
        2. Generate force field parameters using the specific engine (acpype, OpenFF, etc.)
        3. Write files to ``wdir``:
           * ``{ligand.name}.prmtop``: AMBER topology file (required)
           * ``{ligand.name}.inpcrd``: AMBER coordinate file (required)
        4. Read files and store via ``ligand.add_aux_file('prmtop', content)``
           and ``ligand.add_aux_file('inpcrd', content)``
        5. Optionally generate and store additional files (e.g., GROMACS .top file)
        6. Return the ligand object
        
        Optional output files:
        
        * ``{ligand.name}.top``: GROMACS topology (can be added as auxiliary file)
        
        Examples
        --------
        See :meth:`easybfe.smff.gaff.GAFF._parametrize` or
        :meth:`easybfe.smff.openff.OpenFF._parametrize` for reference
        implementations.
        """
        ...