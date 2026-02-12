from pydantic import BaseModel, Field
from .ligand import Ligand


class LigandPerturbation(BaseModel):
    """
    Ligand perturbation data model.
    
    Represents a transformation between two ligands (ligandA -> ligandB) for a given protein,
    including atom mapping, free energy calculations, and analysis data.
    
    Parameters
    ----------
    ligandA : Ligand
        Source ligand (ligand A) in the perturbation.
    ligandB : Ligand
        Target ligand (ligand B) in the perturbation.
    atom_mapping : dict[int, int], default {}
        Atom mapping between ligandA and ligandB. Maps atom indices from ligandA to ligandB.
    ddg_expt : float, default 0.0
        Experimental delta delta G (binding free energy difference) in kcal/mol.
    ddg_expt_std : float, default 0.0
        Standard deviation of experimental delta delta G in kcal/mol.
    ddg_solvent : float, default 0.0
        Calculated delta delta G in solvent phase in kcal/mol.
    ddg_solvent_std : float, default 0.0
        Standard deviation of delta delta G in solvent phase in kcal/mol.
    ddg_complex : float, default 0.0
        Calculated delta delta G in complex phase in kcal/mol.
    ddg_complex_std : float, default 0.0
        Standard deviation of delta delta G in complex phase in kcal/mol.
    ddg_gas : float, default 0.0
        Calculated delta delta G in gas phase in kcal/mol.
    ddg_gas_std : float, default 0.0
        Standard deviation of delta delta G in gas phase in kcal/mol.
    ddg_complex_gas : float, default 0.0
        Calculated delta delta G for complex-gas phase in kcal/mol.
    ddg_complex_gas_std : float, default 0.0
        Standard deviation of delta delta G for complex-gas phase in kcal/mol.
    ddg_total : float, default 0.0
        Total calculated delta delta G in kcal/mol.
    ddg_total_std : float, default 0.0
        Standard deviation of total delta delta G in kcal/mol.
    analysis_data : dict[str, str], default {}
        Additional analysis data stored as key-value pairs (strings).
    """
    ligandA: Ligand = Field(..., description="Source ligand (ligand A) in the perturbation")
    ligandB: Ligand = Field(..., description="Target ligand (ligand B) in the perturbation")
    atom_mapping: dict[int, int] = Field(
        default_factory=dict,
        description="Atom mapping between ligandA and ligandB. Maps atom indices from ligandA to ligandB."
    )
    ddg_expt: float = Field(
        default=0.0,
        description="Experimental delta delta G (binding free energy difference) in kcal/mol"
    )
    ddg_expt_std: float = Field(
        default=0.0,
        description="Standard deviation of experimental delta delta G in kcal/mol"
    )
    ddg_calc: float = Field(
        default=0.0,
        description="Total calculated delta delta G in kcal/mol"
    )
    ddg_calc_std: float = Field(
        default=0.0,
        description="Standard deviation of total delta delta G in kcal/mol"
    )
    ddg_solv: float | None = Field(
        default=None,
        description='ddG solvation in kcal/mol'
    )
    ddg_solv_std: float | None = Field(
        default=None,
        description='ddG solvation standard deviation in kcal/mol'
    )
    dg_solvent: float = Field(
        default=0.0,
        description="Calculated delta delta G in solvent phase in kcal/mol"
    )
    dg_solvent_std: float = Field(
        default=0.0,
        description="Standard deviation of delta delta G in solvent phase in kcal/mol"
    )
    dg_complex: float = Field(
        default=0.0,
        description="Calculated delta delta G in complex phase in kcal/mol"
    )
    dg_complex_std: float = Field(
        default=0.0,
        description="Standard deviation of delta delta G in complex phase in kcal/mol"
    )
    dg_gas: float = Field(
        default=0.0,
        description="Calculated delta delta G in gas phase in kcal/mol"
    )
    dg_gas_std: float = Field(
        default=0.0,
        description="Standard deviation of delta delta G in gas phase in kcal/mol"
    )
    dg_complex_gas: float = Field(
        default=0.0,
        description="Calculated delta delta G for complex-gas phase in kcal/mol"
    )
    dg_complex_gas_std: float = Field(
        default=0.0,
        description="Standard deviation of delta delta G for complex-gas phase in kcal/mol"
    )
    analysis_data: dict[str, str] = Field(
        default_factory=dict,
        description="Additional analysis data stored as key-value pairs (strings)"
    )
