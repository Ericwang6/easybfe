from pathlib import Path
import json
import rich_click as click


@click.group()
def md():
    """Plain MD related commands."""
    pass


@md.command()
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to JSON config file (AmberPlainMDConfig).",
)
@click.option(
    "--ligand",
    "-l",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: ligand input (directory or file).",
)
@click.option(
    "--protein",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: protein PDB path.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config: output directory.",
)
def setup(
    config: Path,
    ligand: Path | None,
    protein: Path | None,
    output_dir: Path | None,
) -> None:
    """Run plain MD setup from a config file. CLI options override config values."""

    from ..config import read_file, AmberPlainMDConfig
    from ..amber.prep_plain_md import setup_plain_md_from_config

    cfg_dict = read_file(str(config))
    if not isinstance(cfg_dict, dict):
        raise click.BadParameter("Config file must contain a dict-like object", param_hint="config")

    # Apply CLI overrides before pydantic validation so validation sees final values
    overrides: dict[str, Path | None] = {}
    if ligand is not None:
        overrides["ligand"] = ligand
    if protein is not None:
        overrides["protein"] = protein
    if output_dir is not None:
        overrides["output_dir"] = output_dir

    for key, value in overrides.items():
        cfg_dict[key] = value

    md_config = AmberPlainMDConfig.model_validate(cfg_dict)
    setup_plain_md_from_config(md_config)


@md.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Plain MD task directory containing config.json and trajectory files.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional analysis config JSON file (PlainMDAnalysisConfig).",
)
@click.option(
    "--prefix",
    "-p",
    type=str,
    default=None,
    help="Production stage name (e.g., '05.prod'). If not provided, inferred from MD config.",
)
@click.option(
    "--basename",
    "-b",
    type=str,
    default=None,
    help="System basename used for files (e.g., 'system'). If not provided, inferred from MD config.",
)
def analyze(
    directory: Path,
    config: Path | None,
    prefix: str | None,
    basename: str | None,
) -> None:
    """Run plain MD analysis workflow."""

    from pydantic import ValidationError

    from ..config import read_file, PlainMDAnalysisConfig
    from ..config.amber.simulation import AmberPlainMDConfig
    from ..analysis.plain_md import run_plain_md_analysis_workflow

    analysis_cfg = None
    if config is not None:
        cfg_dict = read_file(str(config))
        if not isinstance(cfg_dict, dict):
            raise click.BadParameter("Config file must contain a JSON object", param_hint="config")
        analysis_cfg = PlainMDAnalysisConfig.model_validate(cfg_dict)

    # Infer basename/prefix from MD config.json when not provided
    inferred_prefix = prefix
    inferred_basename = basename
    md_cfg_path = directory / "config.json"
    if md_cfg_path.is_file():
        with md_cfg_path.open() as f:
            jdata = json.load(f)
        try:
            md_config = AmberPlainMDConfig.model_validate(jdata)
        except ValidationError:
            cfg = {
                "task_name": jdata.pop("task_name", ""),
                "task_type": jdata.pop("task_type", ""),
                "basename": jdata.pop("basename", ""),
                "simulation": jdata,
            }
            md_config = AmberPlainMDConfig.model_validate(cfg)

        if inferred_basename is None:
            inferred_basename = md_config.simulation.basename
        if inferred_prefix is None:
            inferred_prefix = md_config.simulation.workflow[-1].name

    run_plain_md_analysis_workflow(
        directory=directory,
        config=analysis_cfg,
        prefix=inferred_prefix,
        basename=inferred_basename,
    )

