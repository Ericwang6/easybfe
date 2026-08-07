"""ABFE command line interface (``easybfe abfe ...``)."""
from pathlib import Path

import rich_click as click


@click.group()
def abfe():
    """Absolute binding free energy related commands."""
    pass


@abfe.command()
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to JSON or YAML config file (AmberAbfeConfig).",
)
@click.option(
    "--ligand",
    "-l",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: single ligand directory (mutually exclusive with --ligand-batch).",
)
@click.option(
    "--protein",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: protein PDB path.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config: output directory.",
)
@click.option(
    "--ligand-base",
    "-I",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: ligand_base (parent directory for ligand or ligand_batch paths).",
)
@click.option(
    "--ligand-batch",
    "-L",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: file listing ligand directories (one per line). Mutually exclusive with --ligand.",
)
@click.option(
    "--output-base",
    "-O",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Override config: output_base (required for batch/ligand_batch; "
        "single-ligand writes under output_base/{ligand.name})."
    ),
)
@click.option(
    "--nprocs",
    "-n",
    type=int,
    default=None,
    help="Number of processes for batch runs. Default: auto.",
)
def setup(config: Path, ligand: Path | None, protein: Path | None, output: Path | None, ligand_base: Path | None, ligand_batch: Path | None, output_base: Path | None, nprocs: int | None) -> None:
    """Run ABFE setup from a config file. CLI options override config values."""

    from ..config import read_file
    from .config import AmberAbfeConfig
    from ..amber.prep_ligand_abfe import setup_ligand_abfe_from_config

    if ligand is not None and ligand_batch is not None:
        raise click.UsageError("Cannot set both --ligand and --ligand-batch")

    cfg_dict = read_file(str(config))
    if not isinstance(cfg_dict, dict):
        raise click.BadParameter(
            "Config file must contain a mapping (object) at the root",
            param_hint="config",
        )

    overrides = {}
    if ligand is not None:
        overrides["ligand"] = ligand
        overrides["ligand_batch"] = None
    if protein is not None:
        overrides["protein"] = protein
    if output is not None:
        overrides["output_dir"] = output
    if ligand_base is not None:
        overrides["ligand_base"] = ligand_base
    if ligand_batch is not None:
        with open(ligand_batch) as f:
            paths = [Path(line.strip()) for line in f if line.strip()]
        overrides["ligand_batch"] = paths
        overrides["ligand"] = None
    if output_base is not None:
        overrides["output_base"] = output_base

    # Apply overrides (including None to clear ligand vs ligand_batch)
    for k, v in overrides.items():
        cfg_dict[k] = v

    abfe_config = AmberAbfeConfig.model_validate(cfg_dict)
    setup_ligand_abfe_from_config(abfe_config, num_procs=nprocs)


@abfe.command()
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to JSON or YAML config file (AmberAbfeConfig).",
)
@click.option(
    "--ligand",
    "-l",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: ligand directory or raw ligand file (e.g. SDF).",
)
@click.option(
    "--protein",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: protein PDB path.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config: output directory (<ABFE-DIR>).",
)
def pipeline(config: Path, ligand: Path | None, protein: Path | None, output: Path | None) -> None:
    """Run the full ABFE pipeline (parametrize -> Boresch MD -> setup -> run -> analyze)."""

    from .piepline import ABFE

    runner = ABFE(config, protein=protein, ligand=ligand, output=output)
    runner.run()


@abfe.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=False,
    default=Path("."),
    help="ABFE output directory. Default: current directory.",
)
@click.option(
    "--prod-prefix",
    "-p",
    type=str,
    default="05.prod",
    help="Production run subdirectory name. Default: 05.prod",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=298.15,
    help="Temperature in Kelvin. Default: 298.15",
)
@click.option(
    "--force",
    "-f",
    "force_run",
    is_flag=True,
    default=False,
    help="Re-run MBAR and overwrite result.json even if it exists.",
)
def analyze(
    directory: Path,
    prod_prefix: str,
    temperature: float,
    force_run: bool,
) -> None:
    """Run ABFE analysis (MBAR) and write result.json and convergence plots."""

    from ..analysis.abfe import analyze_abfe

    result = analyze_abfe(
        directory, prod_prefix=prod_prefix, temperature=temperature, force_run=force_run
    )
    if not result:
        click.echo("No result: one or more legs are incomplete.")
        return

    nan = float("nan")
    click.echo(
        f"dG = {result.get('total', nan):.3f} +/- {result.get('total_std', nan):.3f} kcal/mol"
        + ("  [EARLY STOP: pre-production estimate]" if result.get("early_stop") else "")
    )
    convergence = result.get("convergence") or {}
    if convergence:
        click.echo(
            "  convergence: "
            + ("converged" if convergence.get("is_converged") else "NOT converged")
        )
    block = result.get("block_average") or {}
    if block.get("mean") is not None:
        click.echo(f"  block average: {block['mean']:.3f} +/- {block['std']:.3f} kcal/mol")
    for leg, summary in (result.get("legs") or {}).items():
        overlap = (summary.get("overlap") or {}).get("min_adjacent")
        exchange = summary.get("exchange")
        rate = exchange.get("exchange_rate") if exchange else None
        line = f"  {leg:<9} dG = {summary.get('dg', nan):9.3f} +/- {summary.get('dg_std', nan):.3f}"
        line += f"  min overlap = {overlap:.3f}" if overlap is not None else ""
        line += f"  exchange rate = {rate:.3f}" if rate is not None else ""
        click.echo(line)
