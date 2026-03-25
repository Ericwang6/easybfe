import os
import logging
import tempfile
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
from vina import Vina

from .base import BaseDocking
from .embed import constr_embed_with_rdkit, constrained_em_with_protein
from ..cmd import run_command

logger = logging.getLogger(__name__)


class VinaDocking(BaseDocking):
    """Molecular docking with AutoDock Vina via its Python bindings.

    Wraps the :class:`vina.Vina` API to dock, score, and locally optimise
    ligands against a protein receptor.  Ligand conversion between RDKit
    ``Mol`` objects and the PDBQT format required by Vina is handled
    transparently through `meeko`.

    Inherits common initialisation (protein validation, docking-box setup,
    working directory) from :class:`~easybfe.docking.base.BaseDocking`.

    Parameters
    ----------
    protein : os.PathLike
        Path to the protein structure (``.pdb`` or ``.pdbqt``).
    box_center : tuple of float, optional
        ``(x, y, z)`` coordinates of the docking box centre in Angstrom.
        Required unless *ref_mol* is given.
    box_size : tuple of float, optional
        ``(x, y, z)`` dimensions of the docking box in Angstrom.
        Required unless *ref_mol* is given.
    ref_mol : :class:`rdkit.Chem.Mol`, optional
        Reference molecule used to compute *box_center* / *box_size*
        automatically via :func:`~easybfe.docking.base.compute_box_from_coordinates`.
    wdir : os.PathLike, optional
        Working directory for intermediate files (e.g. protein PDBQT).
        A temporary directory is used when *None*.
    protein_prep_exec : str, optional
        Executable for PDB-to-PDBQT protein conversion.  Supported values
        are ``'prepare_receptor'`` (ADFR suite) and ``'obabel'``.
    extra_docking_settings : dict, optional
        Override any of the default docking parameters
        (``num_modes``, ``min_rmsd``, ``energy_range``, ``exhaustiveness``).
    sf_name : str, optional
        Vina scoring-function name (``'vina'``, ``'vinardo'``, or ``'ad4'``).
    cpu : int, optional
        Number of CPUs for Vina (0 = all available).
    seed : int, optional
        Random seed for reproducibility (0 = random).
    verbosity : int, optional
        Vina verbosity level (0 = silent, 1 = normal, 2 = verbose).

    Raises
    ------
    FileNotFoundError
        If *protein* does not exist.
    ValueError
        If neither explicit box parameters nor *ref_mol* are provided.
    """

    DEFAULT_DOCKING_SETTINGS: Dict[str, Any] = {
        "num_modes": 5,
        "min_rmsd": 0.5,
        "energy_range": 5,
        "exhaustiveness": 32,
    }

    def __init__(
        self,
        protein: os.PathLike,
        *,
        box_center: Optional[Tuple[float, float, float]] = None,
        box_size: Optional[Tuple[float, float, float]] = None,
        ref_mol: Optional[Chem.Mol] = None,
        wdir: Optional[os.PathLike] = None,
        protein_prep_exec: str = 'prepare_receptor',
        extra_docking_settings: Optional[Dict[str, Any]] = None,
        sf_name: str = 'vina',
        cpu: int = 0,
        seed: int = 0,
        verbosity: int = 0,
    ):
        super().__init__(
            protein,
            box_center=box_center,
            box_size=box_size,
            ref_mol=ref_mol,
            wdir=wdir,
        )
        self.protein_prep_exec = protein_prep_exec

        self.config: Dict[str, Any] = dict(self.DEFAULT_DOCKING_SETTINGS)
        if extra_docking_settings:
            self.config.update(extra_docking_settings)

        self._vina = Vina(sf_name=sf_name, cpu=cpu, seed=seed, verbosity=verbosity)
        self._prepare_receptor()

    # ------------------------------------------------------------------
    # Protein preparation
    # ------------------------------------------------------------------

    def _prepare_receptor(self):
        """Convert protein to PDBQT (if needed), load into Vina, and
        pre-compute affinity maps for the configured box."""
        if self.protein_input.suffix == '.pdbqt':
            self.protein_pdbqt = self.protein_input
        elif self.protein_input.suffix == '.pdb':
            self.protein_pdbqt = self.wdir / f'{self.protein_input.stem}.pdbqt'
            self.convert_protein_pdb_to_pdbqt(
                self.protein_input,
                self.protein_pdbqt,
                self.protein_prep_exec,
            )
        else:
            raise RuntimeError(
                f"Unsupported protein file format: {self.protein_input.suffix}"
            )

        self._vina.set_receptor(str(self.protein_pdbqt))
        self._vina.compute_vina_maps(center=self.box_center, box_size=self.box_size)
        logger.info("Receptor loaded and Vina maps computed")

    @staticmethod
    def convert_protein_pdb_to_pdbqt(
        protein_pdb: os.PathLike,
        output_path: os.PathLike,
        binary: str = 'prepare_receptor',
    ):
        """Convert a protein PDB file to PDBQT format.

        Parameters
        ----------
        protein_pdb : os.PathLike
            Input PDB file.
        output_path : os.PathLike
            Output PDBQT file.
        binary : str, optional
            Conversion tool -- ``'prepare_receptor'`` (ADFR) or ``'obabel'``.

        Raises
        ------
        NotImplementedError
            If *binary* is not a supported tool.
        """
        name = os.path.basename(binary)
        if name == 'prepare_receptor':
            run_command([binary, '-r', protein_pdb, '-o', output_path,
                         '-A', 'checkhydrogens'])
        elif name == 'obabel':
            run_command([binary, '-ipdb', protein_pdb, '-opdbqt',
                         '-O', output_path, '-xr'])
        else:
            raise NotImplementedError(f'Unsupported protein prep tool: {binary}')

    # ------------------------------------------------------------------
    # Ligand format conversion utilities
    # ------------------------------------------------------------------

    @staticmethod
    def convert_rdkit_to_pdbqt(mol: Chem.Mol) -> str:
        """Convert an RDKit molecule to a PDBQT string via `meeko`.

        Parameters
        ----------
        mol : :class:`rdkit.Chem.Mol`
            Molecule with at least one conformer.

        Returns
        -------
        str
            PDBQT-formatted string.

        Raises
        ------
        ValueError
            If meeko produces an empty PDBQT string.
        """
        molsetup = MoleculePreparation(rigid_macrocycles=True)(mol)[0]
        pdbqt_string = PDBQTWriterLegacy.write_string(molsetup)[0]
        if not pdbqt_string.strip():
            raise ValueError("Meeko produced an empty PDBQT string")
        return pdbqt_string

    @staticmethod
    def pdbqt_string_to_rdmols(pdbqt_string: str) -> List[Chem.Mol]:
        """Convert a (possibly multi-pose) PDBQT string to RDKit molecules.

        Parameters
        ----------
        pdbqt_string : str
            PDBQT content, as returned by :meth:`vina.Vina.poses`.

        Returns
        -------
        list of :class:`rdkit.Chem.Mol`
            One molecule per pose.  Failed conversions are silently dropped.
        """
        pdbqt_mol = PDBQTMolecule(pdbqt_string, skip_typing=True)
        raw = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        out: List[Chem.Mol] = []
        for m in raw:
            if m is None:
                continue
            for cid in range(m.GetNumConformers()):
                single = Chem.Mol(m)
                single.RemoveAllConformers()
                single.AddConformer(Chem.Conformer(m.GetConformer(cid)), assignId=True)
                out.append(single)
        return out

    # ------------------------------------------------------------------
    # Docking / scoring / local optimisation
    # ------------------------------------------------------------------

    def dock(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """Run Vina global-search docking.

        Parameters
        ----------
        mol : :class:`rdkit.Chem.Mol`
            Ligand with at least one 3-D conformer.

        Returns
        -------
        list of :class:`rdkit.Chem.Mol`
            Docked poses ranked by Vina score.
        """
        logger.info("dock: starting global search")
        pdbqt_str = self.convert_rdkit_to_pdbqt(mol)
        self._vina.set_ligand_from_string(pdbqt_str)
        self._vina.dock(
            exhaustiveness=self.config['exhaustiveness'],
            n_poses=self.config['num_modes'],
            min_rmsd=self.config['min_rmsd'],
        )
        poses_pdbqt = self._vina.poses(
            n_poses=self.config['num_modes'],
            energy_range=self.config['energy_range'],
        )
        mols = self.pdbqt_string_to_rdmols(poses_pdbqt)
        logger.info("dock: obtained %d poses", len(mols))
        return mols

    def rescore(self, mol: Chem.Mol) -> float:
        """Score an existing ligand pose without docking.

        Parameters
        ----------
        mol : :class:`rdkit.Chem.Mol`
            Ligand with a 3-D conformer positioned in the binding site.

        Returns
        -------
        float
            Estimated binding free energy in kcal/mol.
        """
        pdbqt_str = self.convert_rdkit_to_pdbqt(mol)
        self._vina.set_ligand_from_string(pdbqt_str)
        energy = self._vina.score()
        return float(energy[0])

    def _vina_optimize(self, mol: Chem.Mol) -> Chem.Mol:
        """Run Vina local BFGS optimisation on an existing pose.

        The PDBQT round-trip (RDKit -> meeko -> Vina -> meeko -> RDKit) may
        reorder atoms.  To keep the output atom ordering consistent with the
        input, a substructure match is used to transfer optimised coordinates
        back onto a copy of the input molecule.

        Parameters
        ----------
        mol : :class:`rdkit.Chem.Mol`
            Ligand with a 3-D conformer.

        Returns
        -------
        :class:`rdkit.Chem.Mol`
            Molecule with optimised coordinates and the same atom ordering
            as the input.

        Raises
        ------
        RuntimeError
            If the optimised pose cannot be converted back.
        """
        mol_in = Chem.Mol(mol)

        pdbqt_str = self.convert_rdkit_to_pdbqt(mol)
        self._vina.set_ligand_from_string(pdbqt_str)
        energy = self._vina.optimize()
        logger.info("Vina optimize: energy = %.3f kcal/mol", float(energy[0]))
        with tempfile.NamedTemporaryFile(suffix='.pdbqt', mode='r', delete=False) as f:
            tmp_path = f.name
        try:
            self._vina.write_pose(tmp_path, overwrite=True)
            with open(tmp_path) as f:
                pose_pdbqt = f.read()
        finally:
            os.unlink(tmp_path)
        mols = self.pdbqt_string_to_rdmols(pose_pdbqt)
        if not mols:
            raise RuntimeError("Failed to convert Vina-optimised pose back to RDKit Mol")
        mol_out = mols[0]

        match = mol_out.GetSubstructMatch(mol_in)
        if not match:
            raise RuntimeError(
                "Substructure match between input and Vina-optimised mol failed; "
                "the PDBQT round-trip may have altered the molecular graph"
            )

        conf_in = mol_in.GetConformer(0)
        conf_out = mol_out.GetConformer(0)
        for idx_in, idx_out in enumerate(match):
            conf_in.SetAtomPosition(idx_in, conf_out.GetAtomPosition(idx_out))
        return mol_in

    def constr_dock(
        self,
        mol: Chem.Mol,
        ref_mol: Chem.Mol,
        mapping: Optional[Dict[int, int]] = None,
        *,
        run_em: bool = True,
        constrain: bool = True,
        restraint_k: float = 10.0,
        output_sdf: Optional[os.PathLike] = None,
    ) -> Chem.Mol:
        """Constrained docking: embed, Vina optimise, and (optionally)
        OpenMM energy-minimise a ligand against *ref_mol*.

        Pipeline:

        1. Clear conformers and generate a new one via constrained
           embedding against *ref_mol*
           (:func:`~easybfe.docking.embed.constr_embed_with_rdkit`).
        2. Locally optimise the pose with Vina
           (:meth:`vina.Vina.optimize`).
        3. *(optional)* Energy-minimise with the full protein using OpenMM
           (:func:`~easybfe.docking.embed.constrained_em_with_protein`).
        4. Compute the heavy-atom RMSD of mapped atoms relative to
           *ref_mol* and store it, together with the Vina score, as
           molecule properties.

        Parameters
        ----------
        mol : :class:`rdkit.Chem.Mol`
            Probe ligand.  Existing conformers are cleared.
        ref_mol : :class:`rdkit.Chem.Mol`
            Reference molecule with a 3-D conformer.
        mapping : dict[int, int], optional
            ``{mol_idx: ref_mol_idx}`` atom mapping.  Auto-generated via
            :class:`~easybfe.mapping.LomapAtomMapper` when *None*.
        run_em : bool
            Whether to run the OpenMM energy-minimisation step with the
            protein.  Requires that the protein was supplied as a ``.pdb``
            file.
        constrain : bool
            If *True* the mapped heavy atoms are frozen during EM
            (mass = 0).  If *False* harmonic restraints with
            *restraint_k* are applied instead.
        restraint_k : float
            Force constant in kcal/(mol * A^2) for harmonic restraints.
            Only used when ``constrain=False``.
        output_sdf : os.PathLike, optional
            If set, write the final pose to this SDF path (parent directories
            are created as needed).  If *None*, no file is written.

        Returns
        -------
        :class:`rdkit.Chem.Mol`
            Optimised ligand with the following molecule properties set:

            * ``vina_score`` -- Vina binding-energy estimate (kcal/mol).
            * ``rmsd`` -- heavy-atom RMSD of mapped atoms vs *ref_mol*
              (Angstrom).
            * ``ff_energy`` -- OpenMM potential energy after EM (kJ/mol),
              only present when *run_em* is *True*.

        Raises
        ------
        RuntimeError
            If *run_em* is *True* but the protein was not provided as a
            ``.pdb`` file.
        """
        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'probe'
        logger.info("constr_dock [%s]: starting", mol_name)

        # 1. Constrained embedding
        mol.RemoveAllConformers()
        mol = Chem.AddHs(Chem.RemoveHs(mol))
        mol, mapping = constr_embed_with_rdkit(mol, ref_mol, mapping=mapping)
        logger.info("constr_dock [%s]: constrained embedding done (%d mapped atoms)",
                     mol_name, len(mapping))

        # 2. Vina local optimisation
        mol = self._vina_optimize(mol)
        logger.info("constr_dock [%s]: Vina local optimisation done", mol_name)

        # 3. Optional OpenMM energy minimisation with protein
        if run_em:
            if self.protein_input.suffix != '.pdb':
                raise RuntimeError(
                    "OpenMM EM requires the original protein input to be a .pdb file, "
                    f"got {self.protein_input.suffix}"
                )
            coord_map: Dict[int, np.ndarray] = {}
            ref_pos = ref_mol.GetConformer().GetPositions()
            for mol_idx, ref_idx in mapping.items():
                coord_map[mol_idx] = ref_pos[ref_idx]
                # if mol.GetAtomWithIdx(mol_idx).GetAtomicNum() != 1:
                    # heavy_coord_map[mol_idx] = ref_pos[ref_idx]

            mol, ff_energy = constrained_em_with_protein(
                mol, self.protein_input, coord_map,
                constrain=constrain, restraint_k=restraint_k,
            )
            mol.SetDoubleProp('ff_energy', ff_energy)
            logger.info("constr_dock [%s]: OpenMM EM done (energy=%.2f kJ/mol)",
                         mol_name, ff_energy)

        # 4. Rescore with Vina
        vina_score = self.rescore(mol)
        mol.SetDoubleProp('vina_score', vina_score)
        logger.info("constr_dock [%s]: Vina rescore = %.3f kcal/mol", mol_name, vina_score)

        # 5. RMSD of mapped heavy atoms
        heavy_mapping = {k: v for k, v in mapping.items()
                         if mol.GetAtomWithIdx(k).GetAtomicNum() != 1}
        prb_idx = list(heavy_mapping.keys())
        ref_idx = list(heavy_mapping.values())
        mol_pos = mol.GetConformer().GetPositions()
        ref_pos = ref_mol.GetConformer().GetPositions()
        rmsd = float(np.sqrt(np.mean(
            np.sum((mol_pos[prb_idx] - ref_pos[ref_idx]) ** 2, axis=1)
        )))
        mol.SetDoubleProp('rmsd', rmsd)
        logger.info("constr_dock [%s]: mapped heavy-atom RMSD = %.3f A", mol_name, rmsd)

        if output_sdf is not None:
            out_path = Path(output_sdf).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with Chem.SDWriter(str(out_path)) as writer:
                writer.write(mol)
            logger.info("constr_dock [%s]: wrote %s", mol_name, out_path)

        return mol
