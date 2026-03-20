from pathlib import Path

import rich_click as click


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
