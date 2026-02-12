"""
SQLAlchemy database models for EasyBFE project.

This module contains the database models for ligands and perturbations.
"""
from __future__ import annotations
from datetime import datetime

from sqlalchemy import Column, String, Float, Integer, Text, UniqueConstraint, DateTime, JSON, ForeignKeyConstraint
from sqlalchemy.orm import DeclarativeBase

from .ligand import Ligand
from .perturbation import LigandPerturbation
from .protein import Protein


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class ProteinDB(Base):
    """
    Database model for protein information.

    Stores protein metadata and the full PDB string so that protein
    information is available directly from the database.
    """
    __tablename__ = 'proteins'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True, index=True)
    desc = Column(String, default='protein')
    pdb_string = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_protein(self) -> Protein:
        """Convert database record to Protein object."""
        return Protein(
            name=self.name,
            desc=self.desc,
            pdb_string=self.pdb_string,
        )

    @classmethod
    def from_protein(cls, protein: Protein) -> 'ProteinDB':
        """Create database record from Protein object."""
        obj = cls()
        obj.update_from_protein(protein)
        return obj
    
    def update_from_protein(self, protein: Protein):
        self.name = protein.name
        self.desc = protein.desc
        self.pdb_string = protein.pdb_string


class LigandDB(Base):
    """
    Database model for ligand information.
    
    Stores all ligand properties from the Ligand object, including molecular
    properties, SMILES, and metadata.
    """
    __tablename__ = 'ligands'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    protein_name = Column(String, nullable=False, index=True)
    ligand_name = Column(String, nullable=False, index=True)
    smiles = Column(String, nullable=False)
    mol_block = Column(Text, default='')
    source = Column(String, default='default')
    mol_weight = Column(Float, default=0.0)
    logp = Column(Float, default=0.0)
    tpsa = Column(Float, default=0.0)
    num_rotatable_bonds = Column(Integer, default=0)
    num_h_donors = Column(Integer, default=0)
    num_h_acceptors = Column(Integer, default=0)
    dG_expt = Column(Float, default=0.0)
    dG_calc = Column(Float, default=0.0)
    auxiliary_files = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        UniqueConstraint('protein_name', 'ligand_name', name='uq_protein_ligand'),
        ForeignKeyConstraint(
            ['protein_name'],
            ['proteins.name'],
            name='fk_ligand_protein'
        ),
    )
    
    def to_ligand(self) -> Ligand:
        """Convert database record to Ligand object."""
        aux_files = self.auxiliary_files if self.auxiliary_files else {}
        ligand = Ligand(
            name=self.ligand_name,
            smiles=self.smiles,
            mol_block=self.mol_block,
            auxiliary_files=aux_files,
            source=self.source,
            mol_weight=self.mol_weight,
            logp=self.logp,
            tpsa=self.tpsa,
            num_rotatable_bonds=self.num_rotatable_bonds,
            num_h_donors=self.num_h_donors,
            num_h_acceptors=self.num_h_acceptors,
            dG_expt=self.dG_expt
        )
        return ligand
    
    @classmethod
    def from_ligand(cls, ligand: Ligand, protein_name: str) -> 'LigandDB':
        """Create database record from Ligand object."""
        obj = cls()
        obj.update_from_ligand(ligand, protein_name)
        return obj
    
    def update_from_ligand(self, ligand: Ligand, protein_name: str = None):
        if protein_name:
            self.protein_name = protein_name
        self.ligand_name = ligand.name
        self.smiles = ligand.smiles
        self.mol_block = ligand.mol_block
        self.source = ligand.source
        self.mol_weight = ligand.mol_weight
        self.logp = ligand.logp
        self.tpsa = ligand.tpsa
        self.num_rotatable_bonds = ligand.num_rotatable_bonds
        self.num_h_donors = ligand.num_h_donors
        self.num_h_acceptors = ligand.num_h_acceptors
        self.dG_expt = ligand.dG_expt
        if self.auxiliary_files is not None:
            self.auxiliary_files.update(ligand.auxiliary_files)
        else:
            self.auxiliary_files = ligand.auxiliary_files


class LigandPerturbationDB(Base):
    """
    Database model for ligand perturbation records.
    
    A perturbation represents a transformation between two ligands for a given protein,
    including atom mapping, free energy calculations, and analysis data.
    """
    __tablename__ = 'ligand_perturbations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    protein_name = Column(String, nullable=False, index=True)
    ligandA_name = Column(String, nullable=False, index=True)
    ligandB_name = Column(String, nullable=False, index=True)
    
    # Atom mapping stored as JSON dict[int, int]
    atom_mapping = Column(JSON, default=dict)
    
    # Experimental delta delta G values
    ddg_expt = Column(Float, default=0.0)
    ddg_expt_std = Column(Float, default=0.0)
    
    # Total calculated delta delta G values
    ddg_calc = Column(Float, default=0.0)
    ddg_calc_std = Column(Float, default=0.0)
    
    # Solvation delta delta G values (optional)
    ddg_solv = Column(Float, nullable=True, default=None)
    ddg_solv_std = Column(Float, nullable=True, default=None)
    
    # Solvent phase delta G values
    dg_solvent = Column(Float, default=0.0)
    dg_solvent_std = Column(Float, default=0.0)
    
    # Complex phase delta G values
    dg_complex = Column(Float, default=0.0)
    dg_complex_std = Column(Float, default=0.0)
    
    # Gas phase delta G values
    dg_gas = Column(Float, default=0.0)
    dg_gas_std = Column(Float, default=0.0)
    
    # Complex-gas phase delta G values
    dg_complex_gas = Column(Float, default=0.0)
    dg_complex_gas_std = Column(Float, default=0.0)
    
    # Analysis data stored as JSON dict[str, str]
    analysis_data = Column(JSON, default=dict)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        UniqueConstraint('protein_name', 'ligandA_name', 'ligandB_name', name='uq_ligand_perturbation'),
        ForeignKeyConstraint(
            ['protein_name'],
            ['proteins.name'],
            name='fk_perturbation_protein'
        ),
        ForeignKeyConstraint(
            ['protein_name', 'ligandA_name'],
            ['ligands.protein_name', 'ligands.ligand_name'],
            name='fk_perturbation_ligandA'
        ),
        ForeignKeyConstraint(
            ['protein_name', 'ligandB_name'],
            ['ligands.protein_name', 'ligands.ligand_name'],
            name='fk_perturbation_ligandB'
        ),
    )
    
    def to_perturbation(self, ligandA: Ligand, ligandB: Ligand) -> LigandPerturbation:
        """
        Convert database record to LigandPerturbation object.
        
        Parameters
        ----------
        ligandA : Ligand
            Ligand A object
        ligandB : Ligand
            Ligand B object
        
        Returns
        -------
        LigandPerturbation
            The perturbation object
        """
        atom_mapping = self.atom_mapping if self.atom_mapping else {}
        analysis_data = self.analysis_data if self.analysis_data else {}
        
        return LigandPerturbation(
            ligandA=ligandA,
            ligandB=ligandB,
            atom_mapping=atom_mapping,
            ddg_expt=self.ddg_expt,
            ddg_expt_std=self.ddg_expt_std,
            ddg_calc=self.ddg_calc,
            ddg_calc_std=self.ddg_calc_std,
            ddg_solv=self.ddg_solv,
            ddg_solv_std=self.ddg_solv_std,
            dg_solvent=self.dg_solvent,
            dg_solvent_std=self.dg_solvent_std,
            dg_complex=self.dg_complex,
            dg_complex_std=self.dg_complex_std,
            dg_gas=self.dg_gas,
            dg_gas_std=self.dg_gas_std,
            dg_complex_gas=self.dg_complex_gas,
            dg_complex_gas_std=self.dg_complex_gas_std,
            analysis_data=analysis_data
        )
    
    @classmethod
    def from_perturbation(cls, perturbation: LigandPerturbation, protein_name: str) -> 'LigandPerturbationDB':
        """
        Create database record from LigandPerturbation object.
        
        Parameters
        ----------
        perturbation : LigandPerturbation
            The perturbation object
        protein_name : str
            Name of the protein
        
        Returns
        -------
        LigandPerturbationDB
            The database record
        """
        return cls(
            protein_name=protein_name,
            ligandA_name=perturbation.ligandA.name,
            ligandB_name=perturbation.ligandB.name,
            atom_mapping=perturbation.atom_mapping,
            ddg_expt=perturbation.ddg_expt,
            ddg_expt_std=perturbation.ddg_expt_std,
            ddg_calc=perturbation.ddg_calc,
            ddg_calc_std=perturbation.ddg_calc_std,
            ddg_solv=perturbation.ddg_solv,
            ddg_solv_std=perturbation.ddg_solv_std,
            dg_solvent=perturbation.dg_solvent,
            dg_solvent_std=perturbation.dg_solvent_std,
            dg_complex=perturbation.dg_complex,
            dg_complex_std=perturbation.dg_complex_std,
            dg_gas=perturbation.dg_gas,
            dg_gas_std=perturbation.dg_gas_std,
            dg_complex_gas=perturbation.dg_complex_gas,
            dg_complex_gas_std=perturbation.dg_complex_gas_std,
            analysis_data=perturbation.analysis_data
        )
