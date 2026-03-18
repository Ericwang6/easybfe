from pathlib import Path

import rich_click as click


@click.group()
def rbfe():
    """Relative binding free energy related commands."""
    pass


@rbfe.command()
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--protein",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: protein PDB path.",
)
@click.option(
    "--ligandA",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: ligand A SDF or directory path.",
)
@click.option(
    "--ligandB",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: ligand B SDF or directory path.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config: output directory.",
)
def setup(
    config: Path,
    protein: Path | None,
    liganda: Path | None,
    ligandb: Path | None,
    output: Path | None,
) -> None:
    """Run RBFE setup from a JSON config file. CLI options override config values."""

    from ..config import read_file
    from ..config.amber.rbfe import AmberLigandRbfeConfig
    from ..amber.prep_ligand_rbfe import setup_ligand_rbfe_from_config

    cfg_dict = read_file(str(config))
    if not isinstance(cfg_dict, dict):
        raise click.BadParameter("Config file must contain a JSON object", param_hint="config")

    overrides = {}
    if liganda is not None:
        overrides["ligandA"] = liganda
    if ligandb is not None:
        overrides["ligandB"] = ligandb
    if protein is not None:
        overrides["protein"] = protein
    if output is not None:
        overrides["output_dir"] = output

    for k, v in overrides.items():
        cfg_dict[k] = v

    rbfe_config = AmberLigandRbfeConfig.model_validate(cfg_dict)
    setup_ligand_rbfe_from_config(rbfe_config)
