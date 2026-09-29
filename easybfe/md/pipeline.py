"""End-to-end plain MD pipeline.

This module exposes :class:`MD`, a single entry point that drives a plain MD run
for a protein, a ligand, or a protein-ligand complex:

1. parameterize the ligand (skipped when an already-parameterized directory or
   ``.ligpack`` is provided, or when there is no ligand);
2. build the solvated system and the MD workflow (``easybfe md setup``);
3. run the workflow locally via the generated ``run.sh``; and
4. analyze the production trajectory (``easybfe md analyze``) and summarize it
   in ``result.json``.

All work happens under a single output directory ``<MD-DIR>``, which is itself a
regular ``easybfe md setup`` directory (so ``easybfe md analyze <MD-DIR>`` works
on it too)::

    <MD-DIR>/ligand/          parameterized (or reloaded) ligand
    <MD-DIR>/protein.pdb      input protein
    <MD-DIR>/system.*         solvated system (prmtop / inpcrd / pdb)
    <MD-DIR>/config.json      resolved AmberPlainMDConfig
    <MD-DIR>/run.sh, 01.em/, ..., <prod>/   the MD workflow and its outputs
    <MD-DIR>/<prod>/prod_processed.{pdb,xtc}, prod_rmsd.{txt,png}, ...
    <MD-DIR>/result.json      analysis summary
    <MD-DIR>/md.log           master log
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import MDPipelineConfig
from ..config import read_file
from ..core import Ligand, Protein
from ..pipeline import attach_log_file, detach_log_file, load_or_parametrize_ligand, run_script


logger = logging.getLogger(__name__)

# GBSA energies come out in kJ/mol.
_KJ_PER_KCAL = 4.184


class MD:
    """Drive a full plain MD run (parameterize -> setup -> run -> analyze).

    Parameters
    ----------
    config : str, pathlib.Path, or easybfe.md.config.MDPipelineConfig
        Pipeline configuration. A path is read (``.yaml``/``.json``) and
        validated as :class:`~easybfe.md.config.MDPipelineConfig`.
    protein : str or pathlib.Path, optional
        Protein PDB path. Overrides ``config.protein`` when given.
    ligand : str or pathlib.Path, optional
        Ligand input: an already-parameterized ligand directory / ``.ligpack``
        or a raw ligand file (e.g. SDF). Overrides ``config.ligand`` when given.
    output : str or pathlib.Path, optional
        Output directory ``<MD-DIR>``. Overrides ``config.output_dir`` when given.

    Notes
    -----
    At least one of protein and ligand is required; which ones are given
    decides whether this is a protein, ligand, or complex simulation. Re-running
    on an ``<MD-DIR>`` that was already set up resumes the workflow instead of
    rebuilding the system.
    """

    def __init__(
        self,
        config: Union[str, os.PathLike, MDPipelineConfig],
        protein: Optional[os.PathLike] = None,
        ligand: Optional[os.PathLike] = None,
        output: Optional[os.PathLike] = None,
    ):
        if isinstance(config, MDPipelineConfig):
            self.config = config.model_copy(deep=True)
        else:
            cfg_dict = read_file(str(config))
            if not isinstance(cfg_dict, dict):
                raise ValueError("Config file must contain a mapping (object) at the root")
            self.config = MDPipelineConfig.model_validate(cfg_dict)

        # CLI/Python overrides take precedence over config values.
        protein_path = protein if protein is not None else self.config.protein
        ligand_input = ligand if ligand is not None else self.config.ligand
        output_dir = output if output is not None else self.config.output_dir

        if protein_path is None and ligand_input is None:
            raise ValueError("At least one of protein and ligand must be provided (via argument or config)")
        if output_dir is None:
            raise ValueError("An output directory must be provided (via argument or config.output_dir)")

        self.protein_path = Path(protein_path).expanduser().resolve() if protein_path is not None else None
        self.ligand_input = Path(ligand_input).expanduser().resolve() if ligand_input is not None else None
        self.root = Path(output_dir).expanduser().resolve()

        self.ligand_dir = self.root / "ligand"
        self.log_file = self.root / "md.log"
        self.result_file = self.root / "result.json"

        self.root.mkdir(parents=True, exist_ok=True)
        self._log_handler: Optional[logging.Handler] = attach_log_file(self.log_file)

        self.ligand: Optional[Ligand] = None

    @property
    def task_type(self) -> str:
        """``'protein'``, ``'ligand'`` or ``'complex'``."""
        if self.ligand_input is None:
            return "protein"
        if self.protein_path is None:
            return "ligand"
        return "complex"

    @property
    def prod_name(self) -> str:
        """Name of the production (last) workflow stage."""
        return self.config.simulation.workflow[-1].name

    def close(self) -> None:
        """Detach and close the pipeline log handler."""
        detach_log_file(self._log_handler)
        self._log_handler = None

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Run the complete pipeline and return the analysis summary."""
        try:
            logger.info("=== MD pipeline start (%s): %s ===", self.task_type, self.root)
            logger.info("Protein: %s", self.protein_path or "-")
            logger.info("Ligand input: %s", self.ligand_input or "-")
            if self._is_set_up():
                logger.info(
                    "Found an existing system and run.sh in %s; resuming the MD workflow "
                    "instead of rebuilding the system.", self.root,
                )
            else:
                self.prepare_ligand()
                self.setup()
            self.run_md()
            result = self.analyze()
            self._log_final_result(result)
            logger.info("=== MD pipeline finished: %s ===", self.root)
            return result
        finally:
            self.close()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def prepare_ligand(self) -> Optional[Ligand]:
        """Load or parameterize the ligand into ``ligand/``; no-op without a ligand."""
        if self.ligand_input is None:
            return None
        self.ligand = load_or_parametrize_ligand(
            self.ligand_input, self.ligand_dir, self.config.ligand_param
        )
        return self.ligand

    def setup(self) -> None:
        """Build the system and the MD workflow, and persist ``config.json``."""
        from ..amber.prep_plain_md import setup_plain_md
        from ..config.amber.simulation import AmberPlainMDConfig

        if self.task_type != "protein" and self.ligand is None:
            raise RuntimeError("prepare_ligand() must run before setup()")

        protein = None
        if self.protein_path is not None:
            protein = Protein.from_pdb(self.protein_path, name=self.protein_path.stem)

        logger.info("Setting up %s MD in %s", self.task_type, self.root)
        setup_plain_md(self.ligand, protein, self.config.simulation, self.root)

        # Same config.json `easybfe md setup` writes, so the analysis below (and a
        # later `easybfe md analyze <MD-DIR>`) sees an ordinary MD directory.
        # Only the analysis fields the user set are passed on, so the rest get
        # their defaults for this system type.
        plain_md_config = AmberPlainMDConfig(
            protein=self.protein_path,
            ligand=self.ligand_dir if self.ligand is not None else None,
            output_dir=self.root,
            task_name=self.config.task_name,
            simulation=self.config.simulation,
            analysis=self.config.analysis.model_dump(exclude_unset=True),
        )
        with open(self.root / "config.json", "w") as f:
            json.dump(plain_md_config.model_dump(mode="json"), f, indent=4)

    def run_md(self) -> None:
        """Run the MD workflow locally; raise if it fails, since nothing after it can run."""
        logger.info("Running MD workflow ...")
        if not run_script(self.root):
            raise RuntimeError(f"MD failed in {self.root}; see {self.root / 'pipeline_run.log'}.")

    def analyze(self) -> dict:
        """Analyze the production trajectory and write ``result.json``."""
        from ..analysis.plain_md import run_plain_md_analysis_workflow

        trajectory = self.root / self.prod_name / f"{self.prod_name}.mdcrd"
        if not trajectory.is_file():
            raise FileNotFoundError(f"Production trajectory not found: {trajectory}")

        logger.info("Running plain-MD analysis on %s ...", trajectory)
        run_plain_md_analysis_workflow(directory=self.root)

        result = self._summarize()
        self.result_file.write_text(json.dumps(result, indent=4))
        logger.info("Wrote analysis summary to %s", self.result_file)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_set_up(self) -> bool:
        basename = self.config.simulation.basename
        return all(
            (self.root / name).is_file()
            for name in ("run.sh", "config.json", f"{basename}.prmtop", f"{basename}.inpcrd")
        )

    def _summarize(self) -> dict:
        """Collect the analysis outputs of the production stage into one dict."""
        prod_dir = self.root / self.prod_name
        with open(self.root / "config.json") as f:
            analysis = json.load(f).get("analysis", {})

        result: dict = {
            "task_type": self.task_type,
            "prod_stage": self.prod_name,
            "trajectory": str(prod_dir / "prod_processed.xtc"),
            "topology": str(prod_dir / "prod_processed.pdb"),
        }

        rmsd_file = prod_dir / "prod_rmsd.txt"
        if rmsd_file.is_file():
            data = np.atleast_2d(np.loadtxt(rmsd_file, comments="#"))
            rmsd = data[:, 1]
            result["rmsd"] = {
                "selection": analysis.get("rmsd_selection"),
                "n_frames": int(len(rmsd)),
                "simulated_ns": float(data[-1, 0]),
                "mean": float(np.mean(rmsd)),
                "std": float(np.std(rmsd)),
                "max": float(np.max(rmsd)),
                "final": float(rmsd[-1]),
                "unit": "angstrom",
            }

        interaction_file = prod_dir / "interaction.csv"
        if interaction_file.is_file():
            result["interaction_csv"] = str(interaction_file)

        gbsa_file = prod_dir / "gbsa.txt"
        if gbsa_file.is_file():
            energies = np.atleast_1d(np.loadtxt(gbsa_file)) / _KJ_PER_KCAL
            result["gbsa"] = {
                "n_frames": int(len(energies)),
                "mean": float(np.mean(energies)),
                "std": float(np.std(energies)),
                "unit": "kcal/mol",
            }
        return result

    def _log_final_result(self, result: dict) -> None:
        logger.info("===== MD RESULT (%s) =====", result.get("task_type"))
        rmsd = result.get("rmsd")
        if rmsd:
            logger.info(
                "  RMSD (%s): mean %.2f +/- %.2f A, max %.2f A, final %.2f A over %.2f ns (%d frames)",
                rmsd["selection"], rmsd["mean"], rmsd["std"], rmsd["max"], rmsd["final"],
                rmsd["simulated_ns"], rmsd["n_frames"],
            )
        gbsa = result.get("gbsa")
        if gbsa:
            logger.info(
                "  GBSA dG_bind: %.2f +/- %.2f kcal/mol (%d frames)",
                gbsa["mean"], gbsa["std"], gbsa["n_frames"],
            )
        if result.get("interaction_csv"):
            logger.info("  Interactions: %s", result["interaction_csv"])
