from typing import Optional, Dict, Literal, Any, Mapping
import warnings
from pydantic import BaseModel, Field, model_validator, ConfigDict


class PlainMDAnalysisConfig(BaseModel):
    """
    Configuration model for MD trajectory analysis workflow.
    
    This model groups all configuration options used in the MD analysis workflow,
    including trajectory processing, RMSD computation, and interaction analysis settings.
    """
    
    model_config = ConfigDict(extra='allow')
    
    # Trajectory processing options
    remove_tmp: bool = Field(
        default=True,
        description='Whether to remove temporary files after processing'
    )
    process_pbc: bool = Field(
        default=False,
        description='Whether to process periodic boundary conditions with MDAnalysis'
    )
    do_alignment: bool = Field(
        default=True,
        description='Whether to align trajectory to reference structure'
    )
    center_selection: str = Field(
        default='',
        description='Selection string for centering the trajectory.'
    )
    output_selection: str = Field(
        default='',
        description='Selection string for output atoms in processed trajectory.'
    )
    align_selection: str = Field(
        default='',
        description='Selection string for alignment reference.'
    )
    
    # RMSD computation options
    rmsd_selection: str = Field(
        default='',
        description='Selection string for RMSD computation.'
    )
    use_symmetry_correction: bool = Field(
        default=True,
        description='Whether to use symmetry correction in RMSD calculation (powered by spyrmsd)'
    )
    heavy_atoms_only: bool = Field(
        default=True,
        description='Whether to only use heavy atoms for RMSD calculation'
    )
    
    # Plotting and naming options
    rmsd_name: str = Field(
        default='', 
        description='Name shown on rmsd plots.'
    )
    interaction_name: str = Field(
        default='', 
        description='Name shown on interaction plots.'
    )
    
    # Interaction analysis options
    interaction_analysis: bool = Field(
        default=True,
        description='Whether to perform interaction analysis'
    )
    use_mpi: bool = Field(
        default=True,
        description='Whether to use MPI for parallel interaction analysis'
    )
    use_strict_hbond: bool = Field(
        default=False,
        description='Whether to use strict hydrogen bond criteria'
    )
    resnr_renum: Dict[str, int] = Field(
        default_factory=dict,
        description='Dictionary for residue number renumbering in interaction analysis'
    )

    # GBSA analysis options
    do_gbsa: bool = Field(
        default=True,
        description='Whether to perform GBSA binding free energy calculation'
    )
    gbsa_igb: int = Field(
        default=2,
        description='GBSA model identifier corresponding to AMBER igb values (1, 2, 5, 7, 8)'
    )
    gbsa_saltcon: float = Field(
        default=0.15,
        description='Salt concentration in M for GBSA calculation'
    )
    gbsa_epsin: float = Field(
        default=4.0,
        description='Solute dielectric constant for GBSA calculation'
    )
    gbsa_epsout: float = Field(
        default=80.0,
        description='Solvent dielectric constant for GBSA calculation'
    )
    gbsa_temperature: float = Field(
        default=298.15,
        description='Temperature in Kelvin for GBSA calculation'
    )


def _infer_selection_from_task_type(task_type: str) -> dict[str, str]:
    data = {}
    data['center_selection'] = 'resname MOL' if task_type == 'ligand' else 'protein'
    data['output_selection'] = 'resname MOL' if task_type == 'ligand' else 'protein or resname MOL'
    data['align_selection'] = 'resname MOL' if task_type == 'ligand' else 'backbone'
    data['rmsd_selection'] = 'backbone' if task_type == 'protein' else 'resname MOL'
    return data


def _infer_plot_names_from_task_name(task_name: str) -> dict[str, str]:
    data = {}
    data['rmsd_name'] = task_name
    data['interaction_name'] = task_name
    return data
