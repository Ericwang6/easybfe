#!/usr/bin/env python
"""
Plot ABFE results vs experimental binding free energies.

This script:
1. Extracts dG.expt from the SDF file
2. Extracts ABFE values from convergence.csv files
3. Subtracts ABFE values by the mean of experimental values
4. Plots the correlation using plot_correlation
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from rdkit import Chem

# Add the repository root to the path to import easybfe
script_path = Path(__file__).resolve()
repo_root = script_path.parents[2]
sys.path.insert(0, str(repo_root))

from easybfe.analysis.plot import plot_correlation


def extract_expt_data(sdf_file: Path) -> dict[str, float]:
    """
    Extract experimental binding free energies from SDF file.
    
    Parameters
    ----------
    sdf_file : Path
        Path to SDF file containing ligands with dG.expt property.
    
    Returns
    -------
    dict[str, float]
        Dictionary mapping ligand name to experimental dG value.
    """
    from easybfe.core.ligand import LigandLoader
    
    expt_data = {}
    loader = LigandLoader()
    ligands = loader.load(str(sdf_file))
    
    for ligand in ligands:
        if ligand.dG_expt != 0.0:  # Only include if dG_expt is set (not default)
            expt_data[ligand.name] = ligand.dG_expt
    
    return expt_data


def extract_abfe_data(abfe_dir: Path) -> dict[str, float]:
    """
    Extract ABFE values from convergence.csv files in subdirectories.
    
    Parameters
    ----------
    abfe_dir : Path
        Directory containing ligand subdirectories with convergence.csv files.
    
    Returns
    -------
    dict[str, float]
        Dictionary mapping ligand name to ABFE value (from last row, Forward column).
    """
    abfe_data = {}
    
    # Find all subdirectories
    for subdir in sorted(abfe_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        
        convergence_file = subdir / 'convergence.csv'
        if not convergence_file.exists():
            continue
        
        try:
            df = pd.read_csv(convergence_file)
            # Get the last row (data_fraction=1.0)
            last_row = df.iloc[4]
            # Use Forward value (should be same as Backward at data_fraction=1.0)
            abfe_value = last_row['Forward']
            abfe_data[subdir.name] = abfe_value
        except Exception as e:
            print(f"Warning: Could not read {convergence_file}: {e}")
            continue
    
    return abfe_data


def main():
    """Main function to extract data and create plot."""
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parents[1]
    
    # Paths
    sdf_file = repo_root / 'tests' / 'data' / 'tyk2_ligands.sdf'
    abfe_dir = script_dir
    
    # Extract data
    print("Extracting experimental data from SDF file...")
    expt_data = extract_expt_data(sdf_file)
    print(f"Found {len(expt_data)} ligands with experimental data")
    
    print("\nExtracting ABFE data from convergence.csv files...")
    abfe_data = extract_abfe_data(abfe_dir)
    print(f"Found {len(abfe_data)} ligands with ABFE data")
    
    # Match ligands
    common_ligands = set(expt_data.keys()) & set(abfe_data.keys())
    print(f"\nFound {len(common_ligands)} ligands with both experimental and ABFE data")
    
    if len(common_ligands) == 0:
        print("Error: No common ligands found!")
        return
    
    # Prepare data arrays
    xdata = []  # Experimental
    ydata = []  # ABFE
    ligand_names = []
    
    for ligand in sorted(common_ligands):
        xdata.append(expt_data[ligand])
        ydata.append(abfe_data[ligand])
        ligand_names.append(ligand)
    
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    
    # Calculate mean of experimental values
    expt_mean = np.mean(list(expt_data.values()))
    print(f"\nMean of experimental values: {expt_mean:.3f} kcal/mol")
    
    # Subtract ABFE values by the mean of experimental values
    ydata_shifted = ydata - (np.mean(ydata) - expt_mean)
    
    print(f"\nPlotting correlation...")
    print(f"Experimental range: {xdata.min():.3f} to {xdata.max():.3f} kcal/mol")
    print(f"ABFE range (after shifting): {ydata_shifted.min():.3f} to {ydata_shifted.max():.3f} kcal/mol")
    
    # Plot
    output_file = script_dir / 'abfe_vs_expt.png'
    ax, stats = plot_correlation(
        xdata, ydata_shifted,
        xlabel=r'$\Delta G_\mathrm{expt}$ (kcal/mol)',
        ylabel=r'$\Delta G_\mathrm{ABFE}$ (kcal/mol)',
        savefig=str(output_file)
    )
    
    print(f"\nPlot saved to: {output_file}")
    print(f"\nStatistics:")
    print(f"  RMSE: {stats['rmse']:.3f} kcal/mol")
    print(f"  R²: {stats['r2']:.3f}")
    print(f"  MUE: {stats['mue']:.3f} kcal/mol")
    print(f"  Slope: {stats['slope']:.3f}")
    print(f"  Intercept: {stats['intercept']:.3f}")
    print(f"  Kendall's τ: {stats['tau']:.3f}")


if __name__ == "__main__":
    main()
