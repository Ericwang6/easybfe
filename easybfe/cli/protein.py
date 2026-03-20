from pathlib import Path

import rich_click as click

from .utils import add_options_from_config
from ..config.protein_prep import ProteinPrepareConfig


@click.group()
def protein():
    """Protein preparation related commands."""
    pass


@protein.command("check")
@click.argument(
    "protein_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--ff",
    "forcefield_files",
    multiple=True,
    help=(
        "Force field XML file. Pass multiple --ff options to provide a "
        "combination. Defaults to amber14-all.xml + amber14/tip3p.xml."
    ),
)
def check(protein_file: Path, forcefield_files: tuple[str, ...]) -> None:
    """Check whether a force field combination can parameterize a protein."""
    from ..protein_prep.utils import check_ff

    ff_files = forcefield_files if forcefield_files else None
    check_ff(protein_file=protein_file, forcefield_files=ff_files)
    click.echo("Force field check passed.")


@protein.command("summary")
@click.argument(
    "protein_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def summary(protein_file: Path) -> None:
    """Print a user-friendly summary of structure quality issues."""
    from ..protein_prep.utils import summary_pdb

    click.echo(summary_pdb(protein_file=protein_file))


@protein.command("fix")
@click.argument(
    "protein_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "-o", "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output PDB file path. Defaults to <input>_fixed.pdb.",
)
@click.option(
    "-w", "--wizard",
    is_flag=True,
    default=False,
    help="Start interactive wizard mode. When set, CLI options are ignored.",
)
@add_options_from_config(ProteinPrepareConfig)
def fix(protein_file: Path, output: Path | None, wizard: bool, **kwargs) -> None:
    """Fix common problems in a protein PDB file.

    In wizard mode (-w/--wizard), interactively prompts for each decision.
    Otherwise, uses the CLI options (or their defaults) to drive the pipeline.
    """
    from ..protein_prep import ProteinFixer

    if output is None:
        output = protein_file.with_stem(protein_file.stem + "_fixed")

    fixer = ProteinFixer(str(protein_file), wizard=wizard)

    if wizard:
        run_kwargs = {}
    else:
        cfg = ProteinPrepareConfig(**{k: v for k, v in kwargs.items() if v is not None})
        run_kwargs = cfg.model_dump(exclude={"forcefield", "res_num_mapping"})

    fixer.run(out=str(output), **run_kwargs)
    click.echo(f"Fixed protein written to {output}")
