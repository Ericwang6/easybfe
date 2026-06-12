from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_angles, calc_bonds, calc_dihedrals
import numpy as np
import pandas as pd


BORESCH_COLUMNS = ("r", "alpha", "theta", "gamma", "beta", "phi")
BORESCH_YLABELS = (
    "r (Angstrom)",
    "alpha (degree)",
    "theta (degree)",
    "gamma (degree)",
    "beta (degree)",
    "phi (degree)",
)

_RST_PATTERN = re.compile(r"&rst\b(.*?)/", re.IGNORECASE | re.DOTALL)
_IAT_PATTERN = re.compile(r"\biat\s*=\s*([0-9,\s]+)", re.IGNORECASE)
_R2_PATTERN = re.compile(
    r"\br2\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)
_CLAMBDA_PATTERN = re.compile(
    r"\bclambda\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)


def lambda_directories(leg_directory: os.PathLike) -> list[Path]:
    """Return lambda directories sorted by their integer suffix."""
    directories = []
    for path in Path(leg_directory).glob("lambda*"):
        suffix = path.name.removeprefix("lambda")
        if path.is_dir() and suffix.isdigit():
            directories.append(path)
    return sorted(directories, key=lambda path: int(path.name.removeprefix("lambda")))


def parse_boresch_restraints(input_path: os.PathLike) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Read the six Boresch atom groups and target values from an Amber input."""
    text = Path(input_path).read_text()
    atom_groups: list[tuple[int, ...]] = []
    targets: list[float] = []

    for block in _RST_PATTERN.findall(text):
        iat_match = _IAT_PATTERN.search(block)
        r2_match = _R2_PATTERN.search(block)
        if iat_match is None or r2_match is None:
            continue
        atoms = tuple(int(value) - 1 for value in iat_match.group(1).split(",") if value.strip())
        if len(atoms) not in (2, 3, 4):
            continue
        atom_groups.append(atoms)
        targets.append(float(r2_match.group(1)))

    if len(atom_groups) < 6:
        raise ValueError(f"Expected six Boresch restraints in {input_path}, found {len(atom_groups)}")

    return atom_groups[-6:], np.asarray(targets[-6:], dtype=float)


def _lambda_value(input_path: Path, lambda_directory: Path) -> float:
    if input_path.is_file():
        match = _CLAMBDA_PATTERN.search(input_path.read_text())
        if match is not None:
            return float(match.group(1))
    return float(lambda_directory.name.removeprefix("lambda"))


def _empty_lambda_frame(leg: str, lambda_value: float, targets: Iterable[float] | None = None) -> pd.DataFrame:
    row: dict[str, float | str] = {"leg": leg, "time": np.nan, "lambda": lambda_value}
    target_values = list(targets) if targets is not None else [np.nan] * 6
    for column, target in zip(BORESCH_COLUMNS, target_values):
        row[column] = np.nan
        row[f"{column}_target"] = target
    return pd.DataFrame([row])


def analyze_boresch_lambda(
    leg: str,
    topology: os.PathLike,
    lambda_directory: os.PathLike,
    prod_prefix: str = "05.prod",
) -> pd.DataFrame:
    """Calculate the six Boresch coordinates for one lambda trajectory."""
    lambda_directory = Path(lambda_directory)
    prod_directory = lambda_directory / prod_prefix
    input_path = prod_directory / f"{prod_prefix}.in"
    trajectory_path = prod_directory / f"{prod_prefix}.mdcrd"
    lambda_value = _lambda_value(input_path, lambda_directory)

    try:
        atom_groups, targets = parse_boresch_restraints(input_path)
    except (OSError, ValueError):
        return _empty_lambda_frame(leg, lambda_value)

    if not trajectory_path.is_file():
        return _empty_lambda_frame(leg, lambda_value, targets)

    try:
        universe = mda.Universe(
            str(topology),
            str(trajectory_path),
            format="NCDF",
        )
    except Exception:
        return _empty_lambda_frame(leg, lambda_value, targets)

    rows = []
    start_time = universe.trajectory[0].time
    for ts in universe.trajectory:
        box = ts.dimensions
        values = []
        for atoms in atom_groups:
            try:
                positions = universe.atoms[list(atoms)].positions
                if len(atoms) == 2:
                    value = calc_bonds(positions[0], positions[1], box=box)
                elif len(atoms) == 3:
                    value = np.degrees(calc_angles(positions[0], positions[1], positions[2], box=box))
                else:
                    value = np.degrees(
                        calc_dihedrals(positions[0], positions[1], positions[2], positions[3], box=box)
                    )
                values.append(float(value))
            except (IndexError, ValueError):
                values.append(np.nan)

        row: dict[str, float | str] = {
            "leg": leg,
            "time": (ts.time - start_time) / 1000.0,
            "lambda": lambda_value,
        }
        for column, value, target in zip(BORESCH_COLUMNS, values, targets):
            row[column] = value
            row[f"{column}_target"] = target
        rows.append(row)

    return pd.DataFrame(rows) if rows else _empty_lambda_frame(leg, lambda_value, targets)


def plot_boresch_coordinates(
    data: pd.DataFrame,
    leg: str,
    save_path: os.PathLike,
    dpi: int = 300,
) -> None:
    """Plot one row per lambda and one column per Boresch coordinate."""
    leg_data = data[data["leg"] == leg]
    lambda_values = sorted(leg_data["lambda"].drop_duplicates())
    if not lambda_values:
        return

    fig, axes = plt.subplots(
        len(lambda_values),
        len(BORESCH_COLUMNS),
        figsize=(21, max(3.0, 2.4 * len(lambda_values))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, lambda_value in enumerate(lambda_values):
        lambda_data = leg_data[leg_data["lambda"] == lambda_value]
        for column_index, (column, ylabel) in enumerate(zip(BORESCH_COLUMNS, BORESCH_YLABELS)):
            ax = axes[row_index, column_index]
            valid = lambda_data[["time", column]].dropna()
            if not valid.empty:
                ax.plot(valid["time"], valid[column], linewidth=1.0)
            targets = lambda_data[f"{column}_target"].dropna()
            if not targets.empty:
                ax.axhline(targets.iloc[0], color="black", linestyle="--", linewidth=1.0)
            if row_index == 0:
                ax.set_title(column)
            if column_index == 0:
                ax.set_ylabel(f"lambda={lambda_value:g}\n{ylabel}")
            if row_index == len(lambda_values) - 1:
                ax.set_xlabel("Time (ns)")
            ax.grid(alpha=0.2)

    fig.suptitle(f"{leg.capitalize()} Boresch restraints")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
