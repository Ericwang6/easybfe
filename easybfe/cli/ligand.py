"""Ligand-related CLI commands (e.g. parameterization)."""
from pathlib import Path

import rich_click as click


@click.group()
def ligand():
    """Ligand loading and parameterization commands."""
    pass


@ligand.command("pargen")
@click.argument(
    "ligand_files",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Ligand file(s) to parameterize (SDF, SMI, CSV, etc.).",
)
@click.option(
    "--output",
    "-o",
    "output_base_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Base directory for per-ligand output subdirectories.",
)
@click.option(
    "--forcefield",
    "-f",
    type=str,
    default="gaff2",
    show_default=True,
    help="Force field name or path (e.g. gaff2, openff-2.1.0, or path to .xml).",
)
@click.option(
    "--charge-method",
    type=str,
    default="bcc",
    show_default=True,
    help="Partial charge assignment method (e.g. bcc, gas).",
)
@click.option(
    "--engine",
    type=str,
    default="",
    help="Explicit engine: acpype, openff, or custom. Auto-detected from forcefield if empty.",
)
@click.option(
    "--raise-errors",
    is_flag=True,
    default=False,
    help="Raise on parametrization errors; otherwise log and skip failed ligands.",
)
@click.option(
    "--nprocs",
    "-n",
    type=int,
    default=-1,
    show_default=True,
    help="Number of parallel processes. -1 = all CPUs, 1 = sequential.",
)
@click.option(
    "--no-name-from-stem",
    is_flag=True,
    default=False,
    help="Do not use filename stem as ligand name (use SDF/CSV name property/column).",
)
@click.option(
    "--only-first",
    is_flag=True,
    default=False,
    help="Only read the first molecule from each SDF/SMI file.",
)
@click.option(
    "--name-prop",
    type=str,
    default="_Name",
    show_default=True,
    help="RDKit property used for molecule name in SDF.",
)
@click.option(
    "--name-col",
    type=str,
    default=None,
    help="Column name for ligand names (CSV/DataFrame).",
)
@click.option(
    "--smi-col",
    type=str,
    default="smiles",
    show_default=True,
    help="Column name for SMILES in CSV/DataFrame.",
)
def pargen(
    ligand_files: tuple[Path, ...],
    output_base_dir: Path,
    forcefield: str,
    charge_method: str,
    engine: str,
    raise_errors: bool,
    nprocs: int,
    no_name_from_stem: bool,
    only_first: bool,
    name_prop: str,
    name_col: str | None,
    smi_col: str,
) -> None:
    """Parameterize one or more ligands with the given force field and write outputs."""
    from ..smff import parametrize_ligands

    source = list(ligand_files) if len(ligand_files) > 1 else ligand_files[0]
    loader_kwargs = {
        "name_from_stem": not no_name_from_stem,
        "only_first": only_first,
        "name_prop": name_prop,
        "smi_col": smi_col,
    }
    if name_col is not None:
        loader_kwargs["name_col"] = name_col

    parametrize_ligands(
        source,
        output_base_dir=output_base_dir,
        forcefield=forcefield,
        charge_method=charge_method,
        engine=engine or "",
        raise_errors=raise_errors,
        nprocs=nprocs,
        **loader_kwargs,
    )
