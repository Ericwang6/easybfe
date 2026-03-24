"""Ligand-related CLI commands (e.g. parameterization)."""
import json
import os
import tempfile
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

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
    "output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for a single ligand (files written directly here).",
)
@click.option(
    "--output-base",
    "-O",
    "output_base",
    type=click.Path(path_type=Path),
    default=None,
    help="Base directory for per-ligand output subdirectories (required for multiple ligands).",
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
    "-c",
    type=str,
    default="bcc",
    show_default=True,
    help="Partial charge assignment method (e.g. bcc, gas, resp).",
)
@click.option(
    "--engine",
    type=str,
    default="",
    help="Explicit engine: acpype, openff, or custom. Auto-detected from forcefield if empty.",
)
@click.option(
    "--resp-engine",
    type=str,
    default="",
    help="Engine for RESP charge calculations (default: qchem). Only used when charge-method starts with 'resp'.",
)
@click.option(
    "--keep-cache",
    is_flag=True,
    default=False,
    help="Keep the intermediate .smff.tmp working directory after parametrization.",
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
    output: Path | None,
    output_base: Path | None,
    forcefield: str,
    charge_method: str,
    engine: str,
    resp_engine: str,
    keep_cache: bool,
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

    if output is None and output_base is None:
        raise click.UsageError("At least one of --output (-o) or --output-base (-O) must be provided.")

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
        output=output,
        output_base=output_base,
        forcefield=forcefield,
        charge_method=charge_method,
        engine=engine or "",
        raise_errors=raise_errors,
        nprocs=nprocs,
        resp_engine=resp_engine,
        keep_cache=keep_cache,
        **loader_kwargs,
    )


@ligand.command("cdock")
@click.argument(
    "probe",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Probe ligand file (SDF etc.); all records are docked against the reference.",
)
@click.option(
    "--ref",
    "-r",
    "ref_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Reference ligand SDF (first record, 3D structure for constrained embedding).",
)
@click.option(
    "--protein",
    "-p",
    "protein",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Receptor structure (.pdb or .pdbqt).",
)
@click.option(
    "--output",
    "-o",
    "output_sdf",
    type=click.Path(path_type=Path),
    default=None,
    help="Output SDF path when the probe file contains a single ligand.",
)
@click.option(
    "--output-dir",
    "-O",
    "output_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for pose SDFs as <dir>/<name>.sdf (required for multiple probes).",
)
@click.option(
    "--box-center",
    "box_center",
    nargs=3,
    type=float,
    default=None,
    metavar="X Y Z",
    help="Docking box centre (Angstrom). Must be used together with --box-size.",
)
@click.option(
    "--box-size",
    "box_size",
    nargs=3,
    type=float,
    default=None,
    metavar="X Y Z",
    help="Docking box size (Angstrom). Must be used together with --box-center.",
)
@click.option(
    "--no-em",
    "no_em",
    is_flag=True,
    default=False,
    help="Skip OpenMM energy minimisation with the protein.",
)
@click.option(
    "--harmonic-restraints",
    "harmonic_restraints",
    is_flag=True,
    default=False,
    help="Use harmonic restraints on mapped atoms during EM instead of freezing them.",
)
@click.option(
    "--restraint-k",
    type=float,
    default=10.0,
    show_default=True,
    help="Harmonic restraint k (kcal/mol/A^2) when --harmonic-restraints is set.",
)
@click.option(
    "--protein-prep-exec",
    type=str,
    default="obabel",
    show_default=True,
    help="Executable for PDB→PDBQT (prepare_receptor or obabel).",
)
@click.option(
    "--mapping-json",
    "mapping_json",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional JSON object {mol_atom_idx: ref_atom_idx} for atom mapping.",
)
@click.option(
    "--sf-name",
    type=str,
    default="vina",
    show_default=True,
    help="Vina scoring function (vina, vinardo, ad4).",
)
@click.option("--cpu", type=int, default=0, show_default=True, help="Vina CPU count (0 = all).")
@click.option("--seed", type=int, default=0, show_default=True, help="Vina random seed (0 = random).")
@click.option(
    "--verbosity",
    type=int,
    default=0,
    show_default=True,
    help="Vina verbosity (0=quiet, 1=normal, 2=verbose).",
)
def cdock(
    probe: Path,
    ref_path: Path,
    protein: Path,
    output_sdf: Path | None,
    output_dir: Path | None,
    box_center: tuple[float, float, float] | None,
    box_size: tuple[float, float, float] | None,
    no_em: bool,
    harmonic_restraints: bool,
    restraint_k: float,
    protein_prep_exec: str,
    mapping_json: Path | None,
    sf_name: str,
    cpu: int,
    seed: int,
    verbosity: int,
) -> None:
    """Constrained local docking: embed probe onto reference, Vina optimise, optional OpenMM EM."""
    from ..core.ligand import LigandLoader
    from ..docking.vina import VinaDocking

    if (box_center is None) != (box_size is None):
        raise click.UsageError("Pass both --box-center and --box-size, or neither.")

    loader = LigandLoader()
    ref_ligands = loader.load(ref_path, only_first=True, name_from_stem=True)
    if not ref_ligands:
        raise click.UsageError(f"No molecule found in {ref_path}")
    ref_mol = ref_ligands[0].get_rdmol()

    probe_ligands = loader.load(probe, only_first=False, name_from_stem=True)
    if not probe_ligands:
        raise click.UsageError(f"No molecule found in {probe}")

    n_probe = len(probe_ligands)
    if output_sdf is None and output_dir is None:
        raise click.UsageError("Provide --output (-o) or --output-dir (-O).")
    if n_probe > 1 and output_dir is None:
        raise click.UsageError("Multiple probe ligands require --output-dir (-O).")
    if output_sdf is not None and n_probe > 1:
        raise click.UsageError(
            "--output (-o) applies to a single probe only; use --output-dir (-O) for multiple."
        )
    if output_sdf is not None and output_dir is not None:
        warnings.warn(
            "Both --output and --output-dir were set; writing under --output-dir and ignoring --output.",
            UserWarning,
            stacklevel=1,
        )

    mapping: dict[int, int] | None = None
    if mapping_json is not None:
        raw: Any = json.loads(mapping_json.read_text())
        if not isinstance(raw, dict):
            raise click.UsageError("--mapping-json must contain a JSON object.")
        mapping = {int(k): int(v) for k, v in raw.items()}

    if box_center is not None and box_size is not None:
        vina_kw: dict[str, Any] = {
            "box_center": tuple(box_center),
            "box_size": tuple(box_size),
            "ref_mol": None,
        }
    else:
        vina_kw = {"ref_mol": ref_mol}

    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if output_dir is not None:
        wdir = output_dir.resolve() / ".tmp"
        wdir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(
            prefix="easybfe_cdock_",
            dir="/tmp" if os.path.isdir("/tmp") else None,
        )
        wdir = Path(tmp_ctx.name)

    try:
        docking = VinaDocking(
            protein,
            wdir=wdir,
            protein_prep_exec=protein_prep_exec,
            sf_name=sf_name,
            cpu=cpu,
            seed=seed,
            verbosity=verbosity,
            **vina_kw,
        )

        multi = n_probe > 1
        for probe_ligand in probe_ligands:
            probe_mol = deepcopy(probe_ligand.get_rdmol())
            if multi:
                click.echo(f"[probe] {probe_ligand.name}")

            if output_dir is not None:
                pose_path = output_dir.resolve() / f"{probe_ligand.name}.sdf"
            else:
                assert output_sdf is not None
                pose_path = output_sdf.resolve()

            result = docking.constr_dock(
                probe_mol,
                ref_mol,
                mapping=mapping,
                run_em=not no_em,
                constrain=not harmonic_restraints,
                restraint_k=restraint_k,
                output_sdf=pose_path,
            )

            score = result.GetDoubleProp("vina_score")
            rmsd = result.GetDoubleProp("rmsd")
            click.echo(f"vina_score: {score:.4f} kcal/mol")
            click.echo(f"rmsd: {rmsd:.4f} Angstrom")
            if result.HasProp("ff_energy"):
                click.echo(f"ff_energy: {result.GetDoubleProp('ff_energy'):.4f} kJ/mol")
            click.echo(f"pose_sdf: {pose_path}")
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
