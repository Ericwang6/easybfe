from typing import Optional, Dict, Literal
import warnings
from pydantic import BaseModel, Field, model_validator


class AnalysisConfig(BaseModel):
    """
    Configuration model for MD trajectory analysis workflow.
    
    This model groups all configuration options used in the MD analysis workflow,
    including trajectory processing, RMSD computation, and interaction analysis settings.
    """
    
    # Task type
    task_type: Literal['protein', 'ligand', 'complex'] = Field(
        default='complex',
        description='Type of MD task: protein-only, ligand-only, or protein-ligand complex'
    )
    
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
    center_selection: Optional[str] = Field(
        default=None,
        description='Selection string for centering the trajectory. If None, will be determined based on task_type'
    )
    output_selection: Optional[str] = Field(
        default=None,
        description='Selection string for output atoms in processed trajectory. If None, will be determined based on task_type'
    )
    align_selection: Optional[str] = Field(
        default=None,
        description='Selection string for alignment reference. If None, will be determined based on task_type'
    )
    
    # RMSD computation options
    rmsd_selection: Optional[str] = Field(
        default=None,
        description='Selection string for RMSD computation. If None, will be determined based on task_type'
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
    rmsd_name: Optional[str] = Field(
        default=None,
        description='Name shown on rmsd plots. If None, will use task_name from directory'
    )
    interaction_name: Optional[str] = Field(
        default=None,
        description='Name shown on interaction plots. If None, will use task_name from directory'
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
    
    @model_validator(mode='after')
    def validate_gbsa(self):
        """
        Validate GBSA-related settings.
        
        Ensures that GBSA analysis is only enabled for complex systems.
        """
        if self.do_gbsa and self.task_type != 'complex':
            self.do_gbsa = False
            warnings.warn("do_gbsa can only be True when task_type is 'complex'.")
        return self

    @model_validator(mode='after')
    def validate_selections(self):
        """
        Set default selection strings based on task_type if not provided.
        """
        # center_selection
        if self.center_selection is None:
            if self.task_type == 'ligand':
                self.center_selection = 'resname MOL'
            else:
                self.center_selection = 'protein'
        
        # output_selection
        if self.output_selection is None:
            if self.task_type == 'ligand':
                self.output_selection = 'resname MOL'
            else:
                self.output_selection = 'protein or resname MOL'
        
        # align_selection
        if self.align_selection is None:
            if self.task_type == 'ligand':
                self.align_selection = 'resname MOL'
            else:
                self.align_selection = 'backbone'
        
        # rmsd_selection
        if self.rmsd_selection is None:
            if self.task_type == 'protein':
                self.rmsd_selection = 'backbone'
            else:
                self.rmsd_selection = 'resname MOL'
        
        return self
