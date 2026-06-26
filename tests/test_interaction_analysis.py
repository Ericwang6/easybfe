from pathlib import Path

import math

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from easybfe.analysis.interaction_plip import (
    analyze_multiple_frames,
    plot_interactions,
)


DATA_DIR = Path(__file__).parent / "data" / "select_rep"
TOPOLOGY = DATA_DIR / "prod_processed.pdb"
TRAJECTORY = DATA_DIR / "prod_processed.xtc"

REQUIRED_COLUMNS = {"ligand_idx", "protein_idx", "dist"}


def _split_frames(tmp_path: Path, n_frames: int = 3):
    """Write the first ``n_frames`` of the test trajectory to PDB files."""
    mda = pytest.importorskip("MDAnalysis")
    u = mda.Universe(str(TOPOLOGY), str(TRAJECTORY))
    n_lig = len(u.select_atoms("resname MOL"))
    pdbs = []
    for i in range(min(n_frames, len(u.trajectory))):
        u.trajectory[i]
        f_pdb = tmp_path / f"{i}.pdb"
        u.atoms.write(str(f_pdb))
        pdbs.append(str(f_pdb))
    return pdbs, n_lig


def test_interaction_report_has_index_and_distance_columns(tmp_path: Path):
    """The aggregated report must expose ligand_idx, protein_idx and dist."""
    pdbs, n_lig = _split_frames(tmp_path)
    out_csv = tmp_path / "interaction.csv"

    df = analyze_multiple_frames(
        pdbs,
        f_csv=str(out_csv),
        use_mpi=False,
        write_xml=False,
        ligand_residue_name="MOL",
    )

    assert out_csv.is_file()
    assert REQUIRED_COLUMNS.issubset(df.columns)
    assert not df.empty

    # Distances are positive and within the typical interaction range.
    dists = df["dist"].dropna().astype(float)
    assert (dists > 0).all()
    assert (dists < 12.0).all()

    # Occupancy ratios are valid fractions.
    assert df["ratio"].between(0.0, 1.0).all()
    assert df["residue_ratio"].between(0.0, 1.0).all()

    # The persisted CSV round-trips with the same required columns.
    reloaded = pd.read_csv(out_csv, index_col=0)
    assert REQUIRED_COLUMNS.issubset(reloaded.columns)


def test_interaction_indices_separate_ligand_and_protein(tmp_path: Path):
    """Single-atom contacts reference ligand atoms vs protein atoms correctly."""
    pdbs, n_lig = _split_frames(tmp_path)

    df = analyze_multiple_frames(
        pdbs,
        use_mpi=False,
        write_xml=False,
        ligand_residue_name="MOL",
    )

    # The ligand (residue MOL) is the first block of atoms in the topology, so
    # its 1-based serial numbers fall within [1, n_lig]; protein atoms are
    # beyond that. Group interactions (salt bridges etc.) store comma-joined
    # lists and are skipped here.
    checked = 0
    for _, row in df.iterrows():
        lig, prot = row["ligand_idx"], row["protein_idx"]
        if "," in str(lig) or "," in str(prot):
            continue
        lig_i, prot_i = int(float(lig)), int(float(prot))
        assert 1 <= lig_i <= n_lig
        assert prot_i > n_lig
        checked += 1
    assert checked > 0


def test_plot_interactions_uses_detailed_report(tmp_path: Path):
    """The detailed atom-level report can be plotted without errors."""
    pdbs, _ = _split_frames(tmp_path)
    df = analyze_multiple_frames(
        pdbs, use_mpi=False, write_xml=False, ligand_residue_name="MOL"
    )
    output = tmp_path / "interaction.png"
    ax = plot_interactions(df, save_path=str(output))
    assert output.is_file()
    plt.close(ax.figure)


def test_plot_interactions_supports_water_bridge(tmp_path: Path):
    data = pd.DataFrame(
        [
            {
                "interaction": "water_bridge",
                "resname": "ASP",
                "resnr": 10,
                "chain": "A",
                "ratio": 0.5,
            }
        ]
    )

    output = tmp_path / "interaction.png"
    ax = plot_interactions(data, save_path=output)

    assert output.is_file()
    plt.close(ax.figure)
