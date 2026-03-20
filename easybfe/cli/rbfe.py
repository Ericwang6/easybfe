from pathlib import Path

import rich_click as click


def _read_ligand_pairs_file(path: Path) -> list[tuple[Path, Path]]:
    """Parse a file of ligand pairs: one pair per line, two fields separated by whitespace.

    Empty lines and lines whose first non-whitespace character is ``#`` are skipped.

    Parameters
    ----------
    path : Path
        Path to the pairs file.

    Returns
    -------
    list[tuple[Path, Path]]
        One ``(ligand_a, ligand_b)`` entry per non-comment data line.

    Raises
    ------
    click.BadParameter
        If a data line does not contain exactly two fields, or no pairs were found.
    """
    pairs: list[tuple[Path, Path]] = []
    text = path.read_text()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise click.BadParameter(
                f"{path}: line {lineno}: expected exactly two whitespace-separated "
                f"ligand path tokens, got {len(parts)}: {raw!r}",
                param_hint="ligand_pairs",
            )
        pairs.append((Path(parts[0]), Path(parts[1])))
    if not pairs:
        raise click.BadParameter(
            f"{path}: no ligand pairs found (file empty or only blanks/comments)",
            param_hint="ligand_pairs",
        )
    return pairs


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
    "-P",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: protein PDB path.",
)
@click.option(
    "--ligandA",
    "-a",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: ligand A SDF or directory path.",
)
@click.option(
    "--ligandB",
    "-b",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: ligand B SDF or directory path.",
)
@click.option(
    "--ligand-base",
    "-I",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Override config: ligand_base (parent directory for ligandA/ligandB or pair paths).",
)
@click.option(
    "--ligand-pairs",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Override config: ligand_pairs from a text file (one pair per line: "
        "two paths separated by whitespace; ``#`` starts a comment line)."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config: output_dir (single-pair when output_base is not used).",
)
@click.option(
    "--output-base",
    "-O",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Override config: output_base (required for batch/ligand_pairs; "
        "single-pair writes under output_base/{ligandA.name}~{ligandB.name})."
    ),
)
def setup(
    config: Path,
    protein: Path | None,
    liganda: Path | None,
    ligandb: Path | None,
    ligand_base: Path | None,
    ligand_pairs: Path | None,
    output: Path | None,
    output_base: Path | None,
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
    if ligand_base is not None:
        overrides["ligand_base"] = ligand_base
    if ligand_pairs is not None:
        overrides["ligand_pairs"] = _read_ligand_pairs_file(ligand_pairs)
    if protein is not None:
        overrides["protein"] = protein
    if output is not None:
        overrides["output_dir"] = output
    if output_base is not None:
        overrides["output_base"] = output_base

    for k, v in overrides.items():
        cfg_dict[k] = v

    rbfe_config = AmberLigandRbfeConfig.model_validate(cfg_dict)
    setup_ligand_rbfe_from_config(rbfe_config)


@rbfe.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="RBFE output directory (contains complex/, solvent/, and optional gas/).",
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
    """Run RBFE analysis (MBAR) and write result.json and convergence plots."""

    from ..analysis.rbfe import analyze_rbfe

    analyze_rbfe(directory, prod_prefix=prod_prefix, temperature=temperature, force_run=force_run)
