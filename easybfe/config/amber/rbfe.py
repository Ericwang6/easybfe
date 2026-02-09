from pathlib import Path
from typing_extensions import Self
from typing import List, Union, Any, Optional, Literal
from pydantic import BaseModel, Field, model_validator
from .simulation import SimulationConfig


class AnalysisConfig(BaseModel):
    temperature: float = 298.15


class LigandRbfeConfig(BaseModel):
    
    submit: bool = None
    submit_gas: bool = None

    submit_header: List[str] = Field(default_factory=list)
    submit_command: str = 'sbatch'

    atom_mapping_method: str = 'kartograf'
    atom_mapping_options: dict[str, Any] = Field(default_factory=dict)
    
    protein_ff: str = 'ff14SB'
    water_ff: str = 'tip3p'
    water_model: Optional[Literal['tip3p', 'spce', 'tip4pew', 'tip5p', 'swm4ndp']] = None
    
    solvent: SimulationConfig
    complex: SimulationConfig
    gas: Union[SimulationConfig, None] = None

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @model_validator(mode='before')
    @classmethod
    def validate_submit_header(cls, data: dict[str, Any]):
        header = data.get('submit_header', [])
        if header and (header[0] != '#/bin/bash' or header[0] != '#/bin/sh'):
            header.insert(0, '#/bin/bash')
        data['sumbmit_header'] = header
        return data
            
    @model_validator(mode='before')
    @classmethod
    def validate_perturbations_before(self, data: dict[str, Any]):
        perturbations = data.get('perturbations', None)
        if perturbations is None:
            return data
        
        if isinstance(perturbations, str) or isinstance(perturbations, Path):
            with open(perturbations) as f:
                perturbations = [line.split() for line in f.read().strip().split('\n')]
            
        perturbations_with_names = []
        for p in perturbations:
            if len(p) == 2:
                p.append(None)
            perturbations_with_names.append(p)
        
        return data
    
    @model_validator(mode='after')
    def validate_perturbations_after(self):
        has_a = self.ligandA_name is not None
        has_b = self.ligandB_name is not None
        has_perts = self.perturbations is not None
        
        if (not has_perts):
            has_a = self.ligandA_name is not None
            has_b = self.ligandB_name is not None
            if has_a and has_b:
                self.perturbations = [[self.ligandA_name, self.ligandB_name, self.pert_name]]
            elif has_a or has_b:
                raise ValueError("Only ligandA or ligandB is provided")
        elif (has_a or has_b):
            raise ValueError("Please only specify ligandA/ligandB if you are *not* setting perturbations directly.")
        
        return self
    
    @classmethod
    def default(cls) -> Self:
        ...
