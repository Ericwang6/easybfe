from __future__ import annotations
from typing import TYPE_CHECKING
import tempfile
import os

import math
import numpy as np
from tqdm import tqdm
import openmm as mm
import openmm.app as app
import openmm.unit as unit
if TYPE_CHECKING:
    from ..core import Protein, Ligand


class GBSARunner:
    """
    GBSA (Generalized Born Surface Area) energy calculator using OpenMM.

    This class computes binding free energies using implicit solvent models
    (GBSA) for protein-ligand complexes. It supports multiple GBSA models
    corresponding to AMBER's igb parameter options.

    Parameters
    ----------
    protein : :class:`easybfe.core.Protein`
        Protein object containing topology and coordinates.
    ligand : :class:`easybfe.core.Ligand`
        Ligand object containing topology, coordinates, and force field XML.
    protein_ffs : list[str], optional
        List of protein force field XML files. Default: ['amber14-all.xml'].
    igb : int, optional
        GBSA model identifier corresponding to AMBER igb values:
        * 1: HCT (Hawkins-Cramer-Truhlar) model
        * 2: OBC1 (Onufriev-Bashford-Case GBOBCI) model (default)
        * 5: OBC2 (Onufriev-Bashford-Case GBOBCII) model
        * 7: GBn model
        * 8: GBn2 model
        Default: 2.
    saltcon : float, optional
        Salt concentration in M. Default: 0.15.
    epsin : float, optional
        Solute dielectric constant. Default: 4.0.
    epsout : float, optional
        Solvent dielectric constant. Default: 80.0.
    temperature : float, optional
        Temperature in Kelvin. Default: 298.15.

    Attributes
    ----------
    ligand_system : :class:`openmm.System`
        OpenMM system for ligand only.
    ligand_ctx : :class:`openmm.Context`
        OpenMM context for ligand only.
    protein_system : :class:`openmm.System`
        OpenMM system for protein only.
    protein_ctx : :class:`openmm.Context`
        OpenMM context for protein only.
    complex_system : :class:`openmm.System`
        OpenMM system for protein-ligand complex.
    complex_ctx : :class:`openmm.Context`
        OpenMM context for protein-ligand complex.

    Examples
    --------
    >>> from easybfe.core import Protein, Ligand
    >>> from easybfe.gbsa import GBSARunner
    >>> runner = GBSARunner(protein, ligand, igb=2)
    >>> binding_energy = runner.compute_single_frame(protein_pos, ligand_pos)
    >>> energies = runner.compute_multiple_frames(protein_positions, ligand_positions)
    """

    def __init__(
        self, 
        protein: Protein, ligand: Ligand,
        protein_ffs: list[str] = ['amber14-all.xml', 'amber14/tip3p.xml'],
        igb: int = 2, saltcon: float = 0.15, epsin: float = 4.0, epsout: float = 80.0,
        temperature = 298.15
    ):
        modeller = app.Modeller(app.Topology(), [])

        ligand_pdb = ligand.to_openmm()
        ligand_top = ligand_pdb.topology
        modeller.add(ligand_top, ligand_pdb.positions)
        
        protein_pdb = protein.to_openmm()
        protein_top = protein_pdb.topology
        modeller.add(protein_top, protein_pdb.positions)

        complex_top = modeller.topology
        complex_pos = modeller.positions

        # Write ligand XML to temporary file
        ligand_xml_fd, ligand_xml = tempfile.mkstemp(suffix='.xml', prefix='ligand_')
        try:
            with os.fdopen(ligand_xml_fd, 'w') as f:
                f.write(ligand.auxiliary_files['xml'])
        except KeyError:
            # File descriptor is already closed by the 'with' statement, just unlink the file
            os.unlink(ligand_xml)
            raise ValueError("ligand.auxiliary_files['xml'] not found. Ligand must be parametrized first.")

        # Map igb parameter to OpenMM implicit solvent XML file
        igb_to_xml = {
            1: 'implicit/hct.xml',
            2: 'implicit/obc1.xml',
            5: 'implicit/obc2.xml',
            7: 'implicit/gbn.xml',
            8: 'implicit/gbn2.xml'
        }
        if igb not in igb_to_xml:
            os.unlink(ligand_xml)
            raise ValueError(f"Unsupported igb value: {igb}. Supported values: {list(igb_to_xml.keys())}")
        obc_xml = igb_to_xml[igb]


        ff = app.ForceField(*protein_ffs, ligand_xml, obc_xml)

        kappa = 367.434915 * math.sqrt(saltcon / epsout / temperature)

        self.ligand_system = ff.createSystem(
            ligand_top, nonbondedMethod=app.NoCutoff, 
            soluteDielectric=epsin, solventDielectric=epsout,
            implicitSolventKappa=kappa/unit.nanometer
        )

        self.ligand_ctx = mm.Context(self.ligand_system, mm.VerletIntegrator(0.001))
        self.protein_system = ff.createSystem(
            protein_top, nonbondedMethod=app.NoCutoff, 
            soluteDielectric=epsin, solventDielectric=epsout,
            implicitSolventKappa=kappa/unit.nanometer
        )
        self.protein_ctx = mm.Context(self.protein_system, mm.VerletIntegrator(0.001))

        self.complex_system = ff.createSystem(
            complex_top, nonbondedMethod=app.NoCutoff, 
            soluteDielectric=epsin, solventDielectric=epsout,
            implicitSolventKappa=kappa/unit.nanometer
        )
        self.complex_ctx = mm.Context(self.complex_system, mm.VerletIntegrator(0.001))
        
        # Store ligand XML path for cleanup
        self._ligand_xml_path = ligand_xml
    
    def __del__(self):
        """Clean up temporary ligand XML file."""
        if hasattr(self, '_ligand_xml_path') and os.path.exists(self._ligand_xml_path):
            try:
                os.unlink(self._ligand_xml_path)
            except OSError:
                pass
    
    def compute_single_frame(self, protein_pos: np.ndarray, ligand_pos: np.ndarray) -> float:
        """
        Compute binding free energy for a single frame.

        Parameters
        ----------
        protein_pos : np.ndarray
            Protein coordinates in Angstroms. Shape: (N_protein_atoms, 3).
        ligand_pos : np.ndarray
            Ligand coordinates in Angstroms. Shape: (N_ligand_atoms, 3).

        Returns
        -------
        float
            Binding free energy in kJ/mol. Calculated as:
            E_complex - E_protein - E_ligand

        Examples
        --------
        >>> binding_energy = runner.compute_single_frame(protein_pos, ligand_pos)
        >>> print(f"Binding energy: {binding_energy:.2f} kJ/mol")
        """
        # Convert from Angstroms to nanometers
        protein_pos = protein_pos / 10
        ligand_pos = ligand_pos / 10

        # Compute complex energy
        complex_pos = np.concatenate((ligand_pos, protein_pos))
        self.complex_ctx.setPositions(complex_pos)
        complex_state = self.complex_ctx.getState(energy=True)
        complex_ene = complex_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        # Compute protein energy
        self.protein_ctx.setPositions(protein_pos)
        protein_state = self.protein_ctx.getState(energy=True)
        protein_ene = protein_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        # Compute ligand energy
        self.ligand_ctx.setPositions(ligand_pos)
        ligand_state = self.ligand_ctx.getState(energy=True)
        ligand_ene = ligand_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        # Binding energy = complex - protein - ligand
        binding_energy = complex_ene - protein_ene - ligand_ene
        return binding_energy
    
    def compute_multiple_frames(
        self, 
        protein_positions: np.ndarray, 
        ligand_positions: np.ndarray,
        progress_bar: bool = True
    ) -> np.ndarray:
        """
        Compute binding free energies for multiple frames.

        Parameters
        ----------
        protein_positions : np.ndarray
            Protein coordinates in Angstroms. Shape: (N_frames, N_protein_atoms, 3).
        ligand_positions : np.ndarray
            Ligand coordinates in Angstroms. Shape: (N_frames, N_ligand_atoms, 3).
        progress_bar : bool, optional
            If True, display a progress bar during computation. Default: True.

        Returns
        -------
        np.ndarray
            Array of binding free energies in kJ/mol. Shape: (N_frames,).

        Raises
        ------
        ValueError
            If input arrays do not have the expected shape (N_frames, N_atoms, 3)
            or if the number of frames does not match between protein and ligand.

        Examples
        --------
        >>> energies = runner.compute_multiple_frames(protein_positions, ligand_positions)
        >>> print(f"Average binding energy: {energies.mean():.2f} kJ/mol")
        >>> # Disable progress bar
        >>> energies = runner.compute_multiple_frames(protein_positions, ligand_positions, progress_bar=False)
        """
        # Validate input shapes
        if protein_positions.ndim != 3 or protein_positions.shape[2] != 3:
            raise ValueError(f"protein_positions must have shape (N_frames, N_atoms, 3), got {protein_positions.shape}")
        if ligand_positions.ndim != 3 or ligand_positions.shape[2] != 3:
            raise ValueError(f"ligand_positions must have shape (N_frames, N_atoms, 3), got {ligand_positions.shape}")
        if protein_positions.shape[0] != ligand_positions.shape[0]:
            raise ValueError(f"Number of frames must match: protein has {protein_positions.shape[0]}, ligand has {ligand_positions.shape[0]}")

        n_frames = protein_positions.shape[0]
        energies = np.zeros(n_frames)

        iterator = tqdm(range(n_frames), desc='Computing GBSA energies', total=n_frames) if progress_bar else range(n_frames)
        for i in iterator:
            energies[i] = self.compute_single_frame(protein_positions[i], ligand_positions[i])

        return energies 






