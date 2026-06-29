"""End-to-end ABFE pipeline.

This module exposes :class:`ABFE`, a single entry point that drives the whole
absolute binding free energy workflow:

1. parameterize the ligand (skipped when an already-parameterized directory is
   provided);
2. when the Boresch algorithm is trajectory-based, run a plain protein-ligand MD
   and select a representative structure together with the Boresch anchors;
3. set up the three ABFE legs (solvent, complex, restraint);
4. run each leg locally; and
5. analyze the result.

All work happens under a single output directory ``<ABFE-DIR>`` with the layout::

    <ABFE-DIR>/ligand/       parameterized (or reloaded) ligand
    <ABFE-DIR>/boresch-md/   plain MD + representative structure (MD-based Boresch only)
    <ABFE-DIR>/abfe/         solvent/ complex/ restraint/ + boresch.dat + result.json
    <ABFE-DIR>/abfe.log      master log
"""
from __future__ import annotations

import logging
import os
import shlex
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

from .config import AmberAbfeConfig
from ..cmd import run_command
from ..config import read_file
from ..core import Ligand, Protein


logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s [%(levelname)s] [PID:%(process)d] [%(name)s]: %(message)s"


class ABFE:
    """Drive the full ABFE workflow for a single protein-ligand system.

    Parameters
    ----------
    config : str, pathlib.Path, or easybfe.abfe.config.AmberAbfeConfig
        Pipeline configuration. A path is read (``.yaml``/``.json``) and
        validated as :class:`~easybfe.abfe.config.AmberAbfeConfig`.
    protein : str or pathlib.Path, optional
        Protein PDB path. Overrides ``config.protein`` when given.
    ligand : str or pathlib.Path, optional
        Ligand input: an already-parameterized ligand directory or a raw ligand
        file (e.g. SDF). Overrides ``config.ligand`` when given.
    output : str or pathlib.Path, optional
        Output directory ``<ABFE-DIR>``. Overrides ``config.output_dir`` when
        given.

    Notes
    -----
    CLI/Python arguments take precedence over the values in ``config``. The MD
    and ABFE runs are executed locally (blocking) via the generated ``run.sh``
    scripts.
    """

    def __init__(
        self,
        config: Union[str, os.PathLike, AmberAbfeConfig],
        protein: Optional[os.PathLike] = None,
        ligand: Optional[os.PathLike] = None,
        output: Optional[os.PathLike] = None,
    ):
        if isinstance(config, AmberAbfeConfig):
            self.config = config.model_copy(deep=True)
        else:
            cfg_dict = read_file(str(config))
            if not isinstance(cfg_dict, dict):
                raise ValueError("Config file must contain a mapping (object) at the root")
            self.config = AmberAbfeConfig.model_validate(cfg_dict)

        # CLI/Python overrides take precedence over config values.
        protein_path = protein if protein is not None else self.config.protein
        ligand_input = ligand if ligand is not None else self.config.ligand
        output_dir = output if output is not None else self.config.output_dir

        if protein_path is None:
            raise ValueError("A protein PDB must be provided (via argument or config.protein)")
        if ligand_input is None:
            raise ValueError("A ligand must be provided (via argument or config.ligand)")
        if output_dir is None:
            raise ValueError("An output directory must be provided (via argument or config.output_dir)")

        self.protein_path = Path(protein_path).expanduser().resolve()
        self.ligand_input = Path(ligand_input).expanduser().resolve()
        self.root = Path(output_dir).expanduser().resolve()

        self.ligand_dir = self.root / "ligand"
        self.boresch_md_dir = self.root / "boresch-md"
        self.abfe_dir = self.root / "abfe"
        self.log_file = self.root / "abfe.log"

        self.root.mkdir(parents=True, exist_ok=True)
        self._log_handler: Optional[RotatingFileHandler] = None
        self._attach_log_handler()

        # Populated during the run.
        self.protein: Protein = Protein.from_pdb(self.protein_path, name=self.protein_path.stem)
        self.ligand: Optional[Ligand] = None
        self.restraint = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _attach_log_handler(self) -> None:
        """Route all ``easybfe`` logging into ``<ABFE-DIR>/abfe.log``."""
        pkg_logger = logging.getLogger("easybfe")
        if pkg_logger.level == logging.NOTSET:
            pkg_logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(str(self.log_file), maxBytes=50 * 1024 * 1024, backupCount=5)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        pkg_logger.addHandler(handler)
        self._log_handler = handler

    def close(self) -> None:
        """Detach and close the pipeline log handler."""
        if self._log_handler is not None:
            logging.getLogger("easybfe").removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Run the complete pipeline and return the ABFE result dictionary."""
        try:
            logger.info("=== ABFE pipeline start: %s ===", self.root)
            logger.info("Protein: %s", self.protein_path)
            logger.info("Ligand input: %s", self.ligand_input)
            self.prepare_ligand()
            if self._need_boresch_md():
                self.run_boresch_md()
            else:
                logger.info(
                    "Boresch algorithm '%s' is single-structure; skipping Boresch MD.",
                    self.config.boresch.algorithm,
                )
            self.setup_abfe()
            if self.config.early_stop_threshold is not None:
                result = self.run_abfe_with_early_stop()
            else:
                self.run_abfe()
                result = self.analyze()
                self._log_final_result(result)
            logger.info("=== ABFE pipeline finished: %s ===", self.root)
            return result
        finally:
            self.close()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def prepare_ligand(self) -> Ligand:
        """Load (directory) or parameterize (file) the ligand into ``ligand/``."""
        if self.ligand_input.is_dir():
            logger.info("Loading already-parameterized ligand from %s", self.ligand_input)
            ligand = Ligand.from_directory(self.ligand_input)
            ligand.dump(self.ligand_dir)
        else:
            from ..smff import parametrize_ligands

            param = self.config.ligand_param
            logger.info(
                "Parameterizing ligand %s (forcefield=%s, charge_method=%s, engine=%s)",
                self.ligand_input, param.forcefield, param.charge_method, param.engine or "auto",
            )
            results = parametrize_ligands(
                str(self.ligand_input),
                output=str(self.ligand_dir),
                forcefield=param.forcefield,
                charge_method=param.charge_method,
                engine=param.engine,
                resp_engine=param.resp_engine,
                raise_errors=True,
                nprocs=1,
                only_first=True,
                name_from_stem=True,
            )
            if not results:
                raise RuntimeError(f"Parameterization produced no ligand for {self.ligand_input}")
            ligand = results[0]

        self.ligand = ligand
        logger.info("Ligand ready: %s (%d atoms)", ligand.name, ligand.get_rdmol().GetNumAtoms())
        return ligand

    def run_boresch_md(self):
        """Run plain protein-ligand MD, then select Boresch anchors + a
        representative protein/ligand pose from the trajectory.

        Updates :attr:`protein` and :attr:`ligand` to the representative pose and
        stores the resulting :class:`~easybfe.boresch.restraint.BoreschRestraint`
        on :attr:`restraint`.
        """
        import json

        from ..amber.prep_plain_md import setup_plain_md
        from ..analysis.plain_md import run_plain_md_analysis_workflow
        from ..boresch import BORESCH_FINDER_REGISTRY
        from ..config.amber.simulation import AmberPlainMDConfig

        assert self.ligand is not None, "prepare_ligand() must run before run_boresch_md()"

        md_cfg = self.config.boresch_md
        logger.info("Setting up Boresch MD (protein + ligand) in %s", self.boresch_md_dir)
        setup_plain_md(self.ligand, self.protein, md_cfg, self.boresch_md_dir)
        # Mirror `easybfe md setup`: persist a config.json next to the run so the
        # subsequent analysis can be driven exactly like `easybfe md analyze`.
        plain_md_config = AmberPlainMDConfig(
            protein=self.protein_path,
            ligand=self.ligand_dir,
            output_dir=self.boresch_md_dir,
            simulation=md_cfg,
        )
        with open(self.boresch_md_dir / "config.json", "w") as f:
            json.dump(plain_md_config.model_dump(mode="json"), f, indent=4)

        logger.info("Running Boresch MD ...")
        self._run_script(self.boresch_md_dir)

        prod_name = md_cfg.workflow[-1].name
        trajectory = self.boresch_md_dir / prod_name / f"{prod_name}.mdcrd"
        if not trajectory.is_file():
            raise FileNotFoundError(f"Boresch MD trajectory not found: {trajectory}")

        # Mirror `easybfe md analyze`: post-process the trajectory and run the
        # RMSD / interaction / GBSA analyses. Processed files land under
        # <boresch-md>/<prod_name>/prod_processed.{pdb,xtc}.
        logger.info("Running plain-MD analysis workflow on Boresch MD trajectory ...")
        run_plain_md_analysis_workflow(directory=self.boresch_md_dir)

        processed_pdb = self.boresch_md_dir / prod_name / "prod_processed.pdb"
        processed_xtc = self.boresch_md_dir / prod_name / "prod_processed.xtc"
        if not processed_pdb.is_file() or not processed_xtc.is_file():
            raise FileNotFoundError(
                "Plain-MD analysis did not produce processed trajectory files "
                f"({processed_pdb}, {processed_xtc})."
            )

        algorithm = self.config.boresch.algorithm
        logger.info("Finding Boresch restraints with algorithm '%s'", algorithm)
        # TODO: the run_plain_md_analysis_workflow will also yield interaction.csv,
        # should pass to the BoreschFinder as well
        finder = BORESCH_FINDER_REGISTRY.create(
            algorithm,
            protein=self.protein,
            ligand=self.ligand,
            wts=tuple(self.config.boresch.rst_wts),
            topology=str(processed_pdb),
            trajectory=str(processed_xtc),
            workdir=str(self.boresch_md_dir),
            **self.config.boresch.options,
        )
        self.restraint = finder.find()

        rep_protein = self.boresch_md_dir / "representative_protein.pdb"
        rep_ligand = self.boresch_md_dir / "representative_ligand.sdf"
        if not rep_protein.is_file() or not rep_ligand.is_file():
            raise FileNotFoundError(
                "Representative structure files were not produced by the Boresch finder "
                f"({rep_protein}, {rep_ligand})."
            )
        logger.info("Adopting representative protein/ligand pose for ABFE setup.")
        self.protein = Protein.from_pdb(rep_protein, name=self.protein.name)
        # TODO: the reload method should be performed with the ``Ligand`` method
        self.ligand = self._reload_ligand_with_representative(self.ligand, rep_ligand)
        self.ligand.dump(self.ligand_dir)
        return self.restraint

    def setup_abfe(self) -> None:
        """Set up the solvent/complex/restraint ABFE legs under ``abfe/``."""
        from ..amber.prep_ligand_abfe import setup_ligand_abfe

        assert self.ligand is not None, "prepare_ligand() must run before setup_abfe()"
        leg_configs = {
            "complex": self.config.complex,
            "solvent": self.config.solvent,
            "restraint": self.config.restraint,
        }
        restraints = self.restraint if self.restraint is not None else self.config.boresch
        logger.info("Setting up ABFE legs in %s", self.abfe_dir)
        setup_ligand_abfe(
            ligand=self.ligand,
            protein=self.protein,
            leg_configs=leg_configs,
            restraints=restraints,
            output_dir=self.abfe_dir,
        )

    def run_abfe(self) -> None:
        """Run each ABFE leg locally (solvent, complex, restraint) in full."""
        for leg in ("solvent", "complex", "restraint"):
            leg_dir = self.abfe_dir / leg
            logger.info("Running ABFE leg '%s' ...", leg)
            self._run_script(leg_dir)

    def run_abfe_with_early_stop(self) -> dict:
        """Run the ABFE legs in two phases with an early-stop check.

        First the pre-production stages (every stage except the last) of all
        three legs are run and the binding free energy is estimated from the
        second-to-last workflow stage. When that estimate is greater than
        :attr:`config.early_stop_threshold` the ligand is treated as a
        weak/non-binder and the production stage is skipped; otherwise the
        production stage is run for every leg and the final result is reported.
        """
        threshold = float(self.config.early_stop_threshold)
        legs = ("solvent", "complex", "restraint")
        workflow = self.config.complex.workflow
        if len(workflow) < 2:
            logger.warning(
                "Workflow has fewer than 2 stages; early stop is not possible. "
                "Running the full workflow instead."
            )
            self.run_abfe()
            result = self.analyze()
            self._log_final_result(result)
            return result

        preprod_prefix = workflow[-2].name
        prod_prefix = workflow[-1].name

        logger.info(
            "Early stop enabled (threshold = %.3f kcal/mol). Running pre-production "
            "stages (through '%s') for all legs first.",
            threshold, preprod_prefix,
        )
        for leg in legs:
            logger.info("Running ABFE leg '%s' pre-production stages ...", leg)
            self._run_script(
                self.abfe_dir / leg, script="run.preprod.sh", done_tag="preprod.done.tag"
            )

        logger.info("Estimating ABFE from pre-production stage '%s' ...", preprod_prefix)
        pre_result = self.analyze(
            prod_prefix=preprod_prefix,
            done_tag="preprod.done.tag",
            run_trajectory_analysis=False,
        )
        pre_dg = pre_result.get("total") if pre_result else None

        if pre_dg is not None and pre_dg > threshold:
            logger.info(
                "EARLY STOP: pre-production dG (%.3f) > threshold (%.3f) kcal/mol. "
                "Ligand treated as a weak/non-binder; skipping production stage '%s' "
                "to save compute.",
                pre_dg, threshold, prod_prefix,
            )
            self._log_final_result(pre_result, early_stopped=True)
            return pre_result

        logger.info(
            "Pre-production dG (%s) <= threshold (%.3f) kcal/mol; running production "
            "stage '%s' for all legs.",
            f"{pre_dg:.3f}" if pre_dg is not None else "n/a", threshold, prod_prefix,
        )
        for leg in legs:
            logger.info("Running ABFE leg '%s' production stage ...", leg)
            self._run_script(self.abfe_dir / leg, script="run.prod.sh", done_tag="done.tag")

        result = self.analyze(prod_prefix=prod_prefix, force_run=True)
        self._log_final_result(result)
        return result

    def analyze(
        self,
        prod_prefix: Optional[str] = None,
        force_run: bool = False,
        done_tag: str = "done.tag",
        run_trajectory_analysis: bool = True,
    ) -> dict:
        """Run MBAR analysis on the completed ABFE directory.

        The free energies are written to ``abfe/result.json`` before the optional
        trajectory/interaction plotting runs. If only that cosmetic post-step
        fails (e.g. resource limits), the already-computed result is still
        returned rather than failing the whole pipeline.

        Parameters
        ----------
        prod_prefix : str, optional
            Production stage name to analyze. Defaults to the last stage of the
            complex workflow.
        force_run : bool, optional
            Recompute even when ``result.json`` already exists.
        done_tag : str, optional
            Per-leg completion tag gating which legs are analyzed (use
            ``"preprod.done.tag"`` for the pre-production estimate).
        run_trajectory_analysis : bool, optional
            Run the (slower) endpoint trajectory / interaction / Boresch
            analyses. Disabled for the quick pre-production estimate.
        """
        import json

        from ..analysis.abfe import analyze_abfe

        if prod_prefix is None:
            prod_prefix = self.config.complex.workflow[-1].name
        temperature = self._temperature()
        result_json = self.abfe_dir / "result.json"
        logger.info(
            "Analyzing ABFE (prod_prefix=%s, T=%.2f K, done_tag=%s)",
            prod_prefix, temperature, done_tag,
        )
        try:
            result = analyze_abfe(
                self.abfe_dir,
                prod_prefix=prod_prefix,
                temperature=temperature,
                force_run=force_run,
                done_tag=done_tag,
                run_trajectory_analysis=run_trajectory_analysis,
            )
        except Exception as exc:
            if result_json.is_file():
                logger.warning(
                    "ABFE free energies were computed, but optional trajectory/interaction "
                    "analysis failed (%s). Returning result from %s.", exc, result_json,
                )
                result = json.loads(result_json.read_text())
            else:
                raise
        if result:
            logger.info(
                "ABFE dG = %.3f +/- %.3f kcal/mol",
                result.get("total", float("nan")), result.get("total_std", float("nan")),
            )
        else:
            logger.warning("ABFE analysis returned no result (legs may be incomplete).")
        return result

    def _log_final_result(self, result: dict, early_stopped: bool = False) -> None:
        """Emit a per-leg and total ABFE summary into the pipeline log."""
        nan = float("nan")
        if not result:
            logger.warning("No final ABFE result to report.")
            return
        tag = " (EARLY STOP; pre-production estimate)" if early_stopped else ""
        logger.info("===== FINAL ABFE RESULT%s =====", tag)
        logger.info(
            "  complex   dG = %.3f +/- %.3f kcal/mol",
            result.get("complex", nan), result.get("complex_std", nan),
        )
        logger.info(
            "  solvent   dG = %.3f +/- %.3f kcal/mol",
            result.get("solvent", nan), result.get("solvent_std", nan),
        )
        logger.info(
            "  restraint dG = %.3f +/- %.3f kcal/mol",
            result.get("restraint", nan), result.get("restraint_std", nan),
        )
        logger.info("  boresch       = %.3f kcal/mol", result.get("boresch", nan))
        logger.info(
            "  TOTAL     dG = %.3f +/- %.3f kcal/mol",
            result.get("total", nan), result.get("total_std", nan),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _need_boresch_md(self) -> bool:
        """True when the configured Boresch algorithm needs a trajectory."""
        return "md" in self.config.boresch.algorithm.lower()

    def _temperature(self) -> float:
        """Production temperature (``temp0``) of the complex leg, else 298.15 K."""
        try:
            temp0 = getattr(self.config.complex.workflow[-1].cntrl, "temp0", None)
            return float(temp0) if temp0 else 298.15
        except Exception:
            return 298.15

    def _reload_ligand_with_representative(self, ligand: Ligand, rep_sdf: os.PathLike) -> Ligand:
        """Return a copy of ``ligand`` with coordinates replaced by ``rep_sdf``.

        The force-field parameters (``xml``/``prmtop``/...) are unchanged; only
        the 3D coordinates in the ``mol_block`` and the ``pdb`` auxiliary file are
        updated to the representative pose. The parameterizer preserves atom
        order between the ligand SDF and PDB, so the representative SDF
        coordinates map onto the PDB atoms one-to-one.
        """
        from rdkit import Chem

        rep_mol = Chem.MolFromMolFile(str(rep_sdf), removeHs=False)
        if rep_mol is None:
            raise ValueError(f"Could not parse representative ligand SDF: {rep_sdf}")
        rep_mol.SetProp("_Name", ligand.name)
        positions = rep_mol.GetConformer().GetPositions()

        new_ligand = ligand.model_copy(deep=True)
        new_ligand.mol_block = Chem.MolToMolBlock(rep_mol)

        pdb_text = new_ligand.auxiliary_files.get("pdb", "")
        if pdb_text:
            new_ligand.auxiliary_files["pdb"] = _replace_pdb_coordinates(pdb_text, positions)
        return new_ligand

    def _run_script(
        self,
        directory: os.PathLike,
        script: str = "run.sh",
        done_tag: str = "done.tag",
    ) -> None:
        """Run ``script`` in ``directory`` (blocking), streaming output to a log.

        The generated scripts are idempotent (they skip when their completion tag
        is already present), so re-running the pipeline resumes from where it
        stopped. The captured stdout/stderr of the script is echoed into the
        pipeline log so it is recorded in ``abfe.log``.

        Parameters
        ----------
        directory : os.PathLike
            Directory containing ``script``.
        script : str, optional
            Script file name to execute (e.g. ``"run.preprod.sh"``).
        done_tag : str, optional
            Completion tag for ``script``. Stale ``running``/``error``/``killed``
            tags sharing the same prefix are cleared before launching.
        """
        directory = Path(directory)
        run_sh = directory / script
        if not run_sh.is_file():
            raise FileNotFoundError(f"{script} not found in {directory}")
        if (directory / done_tag).is_file():
            logger.info("Found %s in %s; skipping.", done_tag, directory)
            return
        # Stale tags (sharing this script's tag prefix) would make the script
        # skip; clear them so a fresh local run can proceed.
        prefix = done_tag[:-len("done.tag")] if done_tag.endswith("done.tag") else ""
        for tag in (f"{prefix}error.tag", f"{prefix}running.tag", f"{prefix}killed.tag"):
            (directory / tag).unlink(missing_ok=True)

        log_path = directory / f"pipeline_{Path(script).stem}.log"
        cmd = ["bash", "-c", f"bash {shlex.quote(script)} > {shlex.quote(str(log_path))} 2>&1"]
        run_command(cmd, cwd=str(directory), raise_error=True)
        logger.info("Finished %s in %s (log: %s)", script, directory, log_path)
        self._log_script_output(directory, script, log_path)

    def _log_script_output(self, directory: Path, script: str, log_path: Path) -> None:
        """Echo a script's captured output into the pipeline log (abfe.log)."""
        try:
            text = Path(log_path).read_text()
        except OSError:
            return
        label = f"{directory.name}/{script}"
        logger.info("----- begin output of %s -----", label)
        for line in text.splitlines():
            logger.info("[%s] %s", label, line)
        logger.info("----- end output of %s -----", label)


def _replace_pdb_coordinates(pdb_text: str, positions) -> str:
    """Overwrite ATOM/HETATM coordinates of ``pdb_text`` with ``positions``.

    ``positions`` is an ``(N, 3)`` array (Angstrom) in the same order as the
    ATOM/HETATM records. Other record types are left untouched.
    """
    lines = pdb_text.splitlines()
    idx = 0
    out = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            if idx >= len(positions):
                raise ValueError(
                    "PDB has more ATOM records than the representative SDF has atoms."
                )
            x, y, z = (float(v) for v in positions[idx])
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            idx += 1
        out.append(line)
    if idx != len(positions):
        raise ValueError(
            f"Atom count mismatch: PDB has {idx} atoms, representative SDF has {len(positions)}."
        )
    return "\n".join(out) + "\n"
