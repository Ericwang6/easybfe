"""Building blocks shared by the one-line pipelines (``abfe pipeline``, ``md pipeline``).

Each pipeline writes everything under one output directory, logs into a single
file there, turns its ligand input into a parameterized ligand, and executes the
generated ``run.sh`` scripts locally (blocking). Those three pieces live here.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .cmd import run_command
from .core import LIGPACK_SUFFIX, Ligand

if TYPE_CHECKING:
    from .smff.config import LigandParamConfig


logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s [%(levelname)s] [PID:%(process)d] [%(name)s]: %(message)s"


def attach_log_file(log_file: os.PathLike) -> logging.Handler:
    """Route all ``easybfe`` logging into ``log_file``; returns the handler to detach later."""
    pkg_logger = logging.getLogger("easybfe")
    if pkg_logger.level == logging.NOTSET:
        pkg_logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(str(log_file), maxBytes=50 * 1024 * 1024, backupCount=5)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    pkg_logger.addHandler(handler)
    return handler


def detach_log_file(handler: Optional[logging.Handler]) -> None:
    """Detach and close a handler returned by :func:`attach_log_file`."""
    if handler is not None:
        logging.getLogger("easybfe").removeHandler(handler)
        handler.close()


def load_or_parametrize_ligand(
    ligand_input: os.PathLike,
    ligand_dir: os.PathLike,
    param: LigandParamConfig,
) -> Ligand:
    """Load (directory / ``.ligpack``) or parameterize (raw file) a ligand into ``ligand_dir``."""
    ligand_input = Path(ligand_input)
    if ligand_input.is_dir() or ligand_input.suffix.lower() == LIGPACK_SUFFIX:
        logger.info("Loading already-parameterized ligand from %s", ligand_input)
        ligand = Ligand.from_path(ligand_input)
        ligand.dump(ligand_dir)
    else:
        from .smff import parametrize_ligands

        logger.info(
            "Parameterizing ligand %s (forcefield=%s, charge_method=%s, engine=%s)",
            ligand_input, param.forcefield, param.charge_method, param.engine or "auto",
        )
        results = parametrize_ligands(
            str(ligand_input),
            output=str(ligand_dir),
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
            raise RuntimeError(f"Parameterization produced no ligand for {ligand_input}")
        ligand = results[0]

    logger.info("Ligand ready: %s (%d atoms)", ligand.name, ligand.get_rdmol().GetNumAtoms())
    return ligand


def read_script_status(directory: os.PathLike) -> dict:
    """Read the ``status.json`` a generated ``run.sh`` writes; ``{}`` when absent."""
    status_file = Path(directory) / "status.json"
    try:
        return json.loads(status_file.read_text())
    except (OSError, ValueError):
        return {}


def run_script(
    directory: os.PathLike,
    script: str = "run.sh",
    args: Optional[list] = None,
    done_tag: str = "done.tag",
) -> bool:
    """Run ``script`` in ``directory`` (blocking), capturing its output.

    The script owns its own tag state machine and resumes at the first stage
    that has not completed, so re-running is safe. ``--force`` is passed
    because the pipeline is the orchestrator here and decides retry policy:
    a run that failed on an earlier invocation should be retried rather than
    blocked by its own ``error.tag``. (Run the script by hand without
    ``--force`` and that guard still applies.)

    Failures are reported, not raised, so the caller decides whether to go on.

    Parameters
    ----------
    directory : os.PathLike
        Directory containing ``script``.
    script : str, optional
        Script file name to execute.
    args : list, optional
        Extra arguments, e.g. ``["--until", "04.pre_prod"]``.
    done_tag : str, optional
        Completion tag that means this phase is already finished.

    Returns
    -------
    bool
        ``True`` when the phase is complete (or was already complete).
    """
    directory = Path(directory)
    run_sh = directory / script
    if not run_sh.is_file():
        raise FileNotFoundError(f"{script} not found in {directory}")
    if (directory / done_tag).is_file():
        logger.info("Found %s in %s; skipping.", done_tag, directory)
        return True

    argv = [script, "--force", *(str(a) for a in (args or []))]
    log_path = directory / f"pipeline_{Path(script).stem}.log"
    shell_cmd = " ".join(shlex.quote(a) for a in ["bash", *argv])
    cmd = ["bash", "-c", f"{shell_cmd} > {shlex.quote(str(log_path))} 2>&1"]
    return_code, _, _ = run_command(cmd, cwd=str(directory), raise_error=False)

    _log_script_output(directory, script, log_path)
    if return_code == 0:
        logger.info("Finished %s in %s (log: %s)", " ".join(argv), directory, log_path)
        return True

    status = read_script_status(directory)
    logger.error(
        "%s failed in %s (exit code %s, stage '%s'): %s",
        " ".join(argv), directory, return_code,
        status.get("stage", "unknown"),
        status.get("error_excerpt") or f"see {log_path}",
    )
    return False


def _log_script_output(directory: Path, script: str, log_path: Path) -> None:
    """Echo a script's captured output into the pipeline log."""
    try:
        text = Path(log_path).read_text()
    except OSError:
        return
    label = f"{directory.name}/{script}"
    logger.info("----- begin output of %s -----", label)
    for line in text.splitlines():
        logger.info("[%s] %s", label, line)
    logger.info("----- end output of %s -----", label)
