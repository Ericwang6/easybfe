"""
Ligand loading and parsing utilities.

This module provides a unified ligand loader that supports multiple input formats
including files (SDF, CSV, SMILES), pandas DataFrames, and RDKit molecule objects.
"""
import io, os
from collections import Counter
from pathlib import Path
from typing import Union, List, Optional, Any, Dict
import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski


logger = logging.getLogger(__name__)


class Ligand(BaseModel):
    """
    Ligand data model.

    Parameters
    ----------
    name : str
        Ligand name/identifier. Must be non-empty and unique within a collection.
    smiles : str
        SMILES string representation of the molecule.
    mol_block : str
        Core 3D structure in SDF mol block format. Contains 3D geometry with
        explicit hydrogens. Empty string if 3D geometry is not available.
    auxiliary_files : Dict[str, str], optional
        Storage for auxiliary files. Dictionary mapping filename to file content
        as string. Can store files like prmtop, inpcrd, itp, etc. Default: empty dict.
    source : str
        Source of the ligand data (e.g., filename, 'DataFrame', 'RDKit').
    mol_weight : float
        Molecular weight (computed from SMILES during validation).
    logp : float
        LogP value (computed from SMILES during validation).
    tpsa : float
        Topological polar surface area (computed from SMILES during validation).
    num_rotatable_bonds : int
        Number of rotatable bonds (computed from SMILES during validation).
    num_h_donors : int
        Number of hydrogen bond donors (computed from SMILES during validation).
    num_h_acceptors : int
        Number of hydrogen bond acceptors (computed from SMILES during validation).

    Examples
    --------
    >>> ligand = Ligand(name="benzene", smiles="c1ccccc1", mol_block="...")
    >>> print(ligand.name)
    benzene
    >>> print(ligand.mol_weight)
    78.11
    >>> ligand.add_aux_file("ligand.prmtop", "content...")
    >>> print(ligand.auxiliary_files)
    {'ligand.prmtop': 'content...'}
    """

    name: str = Field(..., description="Ligand name/identifier")
    smiles: str = Field(..., description="SMILES string representation")
    mol_block: str = Field(default='', description="Core 3D structure in SDF mol block format")
    auxiliary_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Storage for auxiliary files as filename -> content mapping"
    )
    source: str = Field(default='default', description="Source of the ligand data")
    mol_weight: float = Field(default=0.0, description="Molecular weight (computed)")
    logp: float = Field(default=0.0, description="LogP value (computed)")
    tpsa: float = Field(default=0.0, description="Topological polar surface area (computed)")
    num_rotatable_bonds: int = Field(default=0, description="Number of rotatable bonds (computed)")
    num_h_donors: int = Field(default=0, description="Number of hydrogen bond donors (computed)")
    num_h_acceptors: int = Field(default=0, description="Number of hydrogen bond acceptors (computed)")
    charge: int = Field(default=0, description='Net charge of the ligand')
    dG_expt: float = Field(default=0.0, description='Experimental binding free energy to a protein, in kcal/mol')
    
    @model_validator(mode='after')
    def validate_smiles_and_molblock(self):
        """
        Validate SMILES and generate mol_block if empty.

        Normalizes SMILES by removing hydrogens and canonicalizing.
        Generates mol_block from SMILES if mol_block is empty.
        Computes molecular properties (mol_weight, logp, tpsa, etc.).
        """
        m = Chem.MolFromSmiles(self.smiles)
        if m is None:
            raise ValueError(f"Invalid SMILES: {self.smiles}")
        self.smiles = Chem.MolToSmiles(Chem.RemoveHs(m))
        if not self.mol_block:
            # Generate mol_block from the validated molecule
            self.mol_block = Chem.MolToMolBlock(m)
        mol_block_lines = [line.strip('\n') for line in self.mol_block.split('\n')]
        if mol_block_lines[0] != self.name:
            logger.debug(f"Set mol_block name to {self.name}")
            self.mol_block = f'{self.name}\n' + '\n'.join(mol_block_lines[1:])
        
        # Compute molecular properties
        self.mol_weight = Descriptors.MolWt(m)
        self.logp = Crippen.MolLogP(m)
        self.tpsa = Descriptors.TPSA(m)
        self.num_rotatable_bonds = Lipinski.NumRotatableBonds(m)
        self.num_h_donors = Lipinski.NumHDonors(m)
        self.num_h_acceptors = Lipinski.NumHAcceptors(m)
        self.charge = sum([at.GetFormalCharge() for at in m.GetAtoms()])
        
        return self
    
    def get_rdmol(self):
        return Chem.MolFromMolBlock(self.mol_block, removeHs=False)

    def add_aux_file(self, filename: str, content: str):
        """
        Add an auxiliary file to the ligand.

        Parameters
        ----------
        filename : str
            Name of the auxiliary file (e.g., 'ligand.prmtop', 'ligand.inpcrd').
        content : str
            Content of the file as a string.

        Examples
        --------
        >>> ligand.add_aux_file("ligand.prmtop", prmtop_content)
        >>> ligand.add_aux_file("ligand.inpcrd", inpcrd_content)
        """
        self.auxiliary_files[filename] = content
    
    def check_aux_file(self, filename: str):
        assert self.auxiliary_files.get(filename, ''), f'{filename} not set!'
    
    def dump(self, dirname: str):
        dirname = Path(dirname).expanduser().resolve()
        dirname.mkdir(exist_ok=True, parents=True)
        with Chem.SDWriter(os.path.join(dirname, f'{self.name}.sdf')) as w:
            w.write(self.get_rdmol())
        for f, content in self.auxiliary_files.items():
            with open(os.path.join(dirname, f'{self.name}.{f}'), 'w' ) as f:
                f.write(content)
    
    def embed(self, return_new: bool = True):
        mol = self.get_rdmol()
        if _need_3d(mol):
            mol = Chem.AddHs(mol)
            logger.warning("No 3D conformer found. Generating 3D coordinates for molecule")
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
        elif sum([at.GetNumImplicitHs() for at in mol.GetAtoms()]) > 0:
            # Check if molecule has explicit hydrogens, if not add them
            mol = Chem.AddHs(mol, addCoords=True)
        
        mol.SetProp('_Name', self.name)
        mol_block = Chem.MolToMolBlock(mol)
        if return_new:
            data_dict = self.model_dump()
            data_dict['mol_block'] = mol_block
            return self.model_copy(update={"mol_block": mol_block})
        else:
            self.mol_block = mol_block
            return self

    @classmethod
    def from_directory(cls, directory: os.PathLike):
        directory = Path(directory).expanduser().resolve()
        stem = Path(directory).stem
        loader = LigandLoader()
        ligand = loader.load(directory / f'{stem}.sdf', only_first=True, use_stem_as_name=True)[0]
        ligand.source = f'Init from {directory}'
        for file in directory.glob(f'{stem}.*'):
            if file.suffix in ['.sdf', '.png']:
                continue
            ligand.auxiliary_files[file.suffix[1:]] = file.read_text()
        return ligand
    
    @classmethod
    def from_file(cls, src: os.PathLike, **kwargs):
        loader = LigandLoader()
        ligands = loader.load(src, **kwargs)
        if len(ligands) > 1:
            logger.warning(f"Load {len(ligands)} molecules from {src}. Only the first will be returned.")
        return ligands[0]
          

def _need_3d(mol):
    needs_3d = False
    try:
        conf = mol.GetConformer()
        if conf is None or not conf.Is3D():
            needs_3d = True
    except (ValueError, RuntimeError):
        # No conformer exists
        needs_3d = True
    return needs_3d


class LigandLoader:
    """
    Unified ligand loader supporting file upload (Web) and native objects (Python API).

    This class handles parsing of various input formats and validates ligand data.
    It supports:

    * SDF files (single or multiple molecules)
    * CSV files with name and SMILES columns
    * SMILES files (.smi, .smiles)
    * Pandas DataFrames
    * Lists of RDKit molecule objects

    Processing logic:

    1. Parse various input formats
    2. Validate that names are non-empty
    3. Validate that names are unique within the collection

    Examples
    --------
    >>> loader = LigandLoader()
    >>> # Load from SDF file
    >>> ligands = loader.load("ligands.sdf")
    >>> # Load from DataFrame
    >>> df = pd.DataFrame({"Name": ["benzene"], "SMILES": ["c1ccccc1"]})
    >>> ligands = loader.load(df, name_col="Name", smi_col="SMILES")
    >>> # Load from RDKit molecules
    >>> mols = [Chem.MolFromSmiles("c1ccccc1")]
    >>> ligands = loader.load(mols)
    """

    def __init__(self):
        """Initialize the ligand loader."""

    def load(self, source: Any, enforce_unique_name: bool = True, **kwargs) -> List[Ligand]:
        """
        Unified entry method that dispatches parsing logic based on source type.

        Parameters
        ----------
        source : Any
            Input source. Supported types:
            * :class:`pandas.DataFrame`: DataFrame with name and SMILES columns
            * :class:`list` of :class:`rdkit.Chem.Mol`: List of RDKit molecules
            * :class:`str`, :class:`pathlib.Path`, or :class:`list` of paths: File paths
        enforce_unique_name : bool, optional
            If True (default), validate that all ligand names are unique.
            If False, skip uniqueness validation.
        **kwargs
            Additional keyword arguments:
            * name_col (str): Column name for ligand names (DataFrame/CSV)
            * smi_col (str): Column name for SMILES (DataFrame/CSV)
            * name_from_stem (bool): Use filename stem as name (SDF)
            * only_first (bool): Only read first molecule from SDF file

        Returns
        -------
        List[Ligand]
            List of validated Ligand objects.

        Raises
        ------
        ValueError
            If input type is not supported or duplicate names are found (when
            enforce_unique_name=True).

        Examples
        --------
        >>> loader = LigandLoader()
        >>> ligands = loader.load("ligands.sdf")
        >>> ligands = loader.load(df, name_col="Name", smi_col="SMILES")
        >>> ligands = loader.load([mol1, mol2], enforce_unique_name=False)
        """
        # Dispatch parsing logic based on input type
        if isinstance(source, pd.DataFrame):
            name_col = kwargs.get("name_col")
            smi_col = kwargs.get("smi_col")
            if name_col is None or smi_col is None:
                raise ValueError("name_col and smi_col must be provided for DataFrame")
            items = self._from_dataframe(source, name_col, smi_col)
        elif isinstance(source, list) and len(source) > 0 and isinstance(source[0], Chem.Mol):
            items = self._from_rdkit_mol_list(source)
        elif isinstance(source, (str, Path, list)):
            # Handle file paths or list of file paths (SDF, CSV, SMI)
            items = self._from_files(source, **kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(source)}")

        # Convert to Ligand models and validate uniqueness
        ligands = [Ligand(**item) for item in items]
        if enforce_unique_name:
            self._validate_uniqueness(ligands)
        return ligands

    # --- Internal parsing logic (private methods) ---

    def _from_files(
        self, paths: Union[str, Path, List[Union[str, Path]]], **kwargs
    ) -> List[dict]:
        """
        Parse ligands from file paths.

        Parameters
        ----------
        paths : str, Path, or List[Union[str, Path]]
            Single file path or list of file paths.
            When a single path is provided, name_from_stem defaults to False.
            When a list of paths is provided, name_from_stem can be set via kwargs.
        **kwargs
            Additional arguments passed to format-specific parsers:
            * name_from_stem (bool): Use filename stem as name (SDF, only for list of paths)
            * only_first (bool): Only read first molecule from SDF file
            * name_col (str): Column name for ligand names (CSV)
            * smi_col (str): Column name for SMILES (CSV)

        Returns
        -------
        List[dict]
            List of dictionaries with 'name', 'smiles', 'mol_block', and 'source' keys.

        Raises
        ------
        FileNotFoundError
            If a file path does not exist.
        ValueError
            If file format is not supported.
        """
        if isinstance(paths, (str, Path)):
            path_list = [paths]
            name_from_stem = kwargs.get("name_from_stem", False)
        else:
            path_list = paths 
            name_from_stem = False
            
        all_items = []

        for p in path_list:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            
            content = p.read_bytes()  # Read bytes to support Web uploads
            
            if p.suffix.lower() == ".sdf":
                only_first = kwargs.get("only_first", False)
                all_items.extend(self._parse_sdf(content, p.name, name_from_stem, only_first))
            elif p.suffix.lower() == ".csv":
                name_col = kwargs.get("name_col")
                smi_col = kwargs.get("smi_col")
                if name_col is None or smi_col is None:
                    raise ValueError("name_col and smi_col must be provided for CSV files")
                df = pd.read_csv(io.BytesIO(content))
                all_items.extend(self._from_dataframe(df, name_col, smi_col, p.name))
            
            elif p.suffix.lower() in [".smi", ".smiles"]:
                all_items.extend(self._parse_smi(content, p.name))
            else:
                raise ValueError(f"Unsupported file format: {p.suffix}")

        return all_items

    def _parse_sdf(
        self, content: bytes, filename: str, use_stem: bool, only_first: bool
    ) -> List[dict]:
        """
        Parse ligands from SDF file content with 3D geometry.

        Parameters
        ----------
        content : bytes
            SDF file content as bytes.
        filename : str
            Original filename (used for source attribute).
        use_stem : bool
            If True, use filename stem as ligand name; otherwise use _Name property.
        only_first : bool
            If True, only parse the first molecule in the file.

        Returns
        -------
        List[dict]
            List of dictionaries with 'name', 'smiles', 'mol_block', and 'source' keys.

        Raises
        ------
        ValueError
            If molecule cannot be parsed, has invalid name, or cannot be converted to SMILES.
        RuntimeError
            If mol block generation fails.

        Notes
        -----
        Molecules are loaded with explicit hydrogens preserved (removeHs=False).
        Mol block is generated from the molecule with 3D coordinates.
        """
        suppl = Chem.SDMolSupplier()
        suppl.SetData(content, removeHs=False)
        mols = [next(suppl)] if only_first else [m for m in suppl]
        for i, m in enumerate(mols):
            if m is None:
                raise ValueError(f"Fail to parse molecule {i} in {filename}")
        names = [Path(filename).stem for m in mols] if use_stem else []
        items = self._from_rdkit_mol_list(mols, names=names)
        for i in range(len(items)):
            items[i]['source'] = filename
        return items

    def _parse_smi(self, content: bytes, filename: str) -> List[dict]:
        """
        Parse ligands from SMILES file content.

        Parameters
        ----------
        content : bytes
            SMILES file content as bytes.
        filename : str
            Original filename (used for source attribute).

        Returns
        -------
        List[dict]
            List of dictionaries with 'name', 'smiles', 'mol_block', and 'source' keys.
            SMILES validation and mol_block generation are handled by Ligand validator.

        Notes
        -----
        SMILES file format: one line per molecule, format is "SMILES name" or just "SMILES".
        If no name is provided, a default name based on line number is used.
        SMILES validation and mol_block generation (if empty) are handled automatically
        by the Ligand model validator.
        """
        text = content.decode("utf-8")
        items = []
        
        for line_num, line in enumerate(text.strip().split("\n"), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()  
            smiles = parts[0]
            
            if len(parts) > 1:
                name = parts[1].strip()
            else:
                name = f"{Path(filename).stem}_line_{line_num}"
            
            # Let validator handle SMILES validation and mol_block generation
            items.append({
                "name": name,
                "smiles": smiles,
                "mol_block": "",
                "source": filename
            })
        
        return items
    
    def _from_rdkit_mol(self, mol: Chem.Mol, name_prop: str = '_Name', name: str | None = None):
        smiles = Chem.MolToSmiles(mol)
        mol_block = Chem.MolToMolBlock(mol)
        if mol.HasProp('dG.expt'):
            dG_expt = mol.GetDoubleProp('dG.expt')
        elif mol.HasProp('affinity.expt'):
            dG_expt = 298.15 * 8.314 * np.log(mol.GetDoubleProp('affinity.expt') * 1e-6) / 1000 / 4.184
        else:
            dG_expt = 0.0

        if not name:
            if mol.HasProp(name_prop):
                name = mol.GetProp(name_prop)
            else:
                name = ''
        return {"name": name, "smiles": smiles, "mol_block": mol_block, "dG_expt": dG_expt}

    def _from_rdkit_mol_list(self, mols: List[Chem.Mol], name_prop: str = '_Name', names: list[str] = list()) -> List[dict]:
        """
        Parse ligands from a list of RDKit molecule objects.

        Parameters
        ----------
        mols : List[Chem.Mol]
            List of RDKit molecule objects.

        Returns
        -------
        List[dict]
            List of dictionaries with 'name', 'smiles', 'mol_block', and 'source' keys.

        Raises
        ------
        ValueError
            If molecule is None or cannot be converted to SMILES.

        Notes
        -----
        Molecules without _Name property are assigned default names (mol_0, mol_1, ...).
        Mol block is generated from the molecule using MolToMolBlock.
        """
        items = []
        for i, m in enumerate(mols):
            items.append(self._from_rdkit_mol(m, name_prop=name_prop))
        if names:
            for i, (item, name) in enumerate(zip(items, names, strict=True)):
                items[i]['name'] = name
        return items

    def _from_dataframe(
        self, df: pd.DataFrame, name_col: str, smi_col: str, source: str = "DataFrame"
    ) -> List[dict]:
        """
        Parse ligands from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ligand data.
        name_col : str
            Column name containing ligand names.
        smi_col : str
            Column name containing SMILES strings.
        source : str, optional
            Source identifier (default: "DataFrame"). Used for CSV files to pass filename.

        Returns
        -------
        List[dict]
            List of dictionaries with 'name', 'smiles', 'mol_block', and 'source' keys.
            SMILES validation and mol_block generation are handled by Ligand validator.

        Raises
        ------
        KeyError
            If specified columns are not found in the DataFrame.

        Notes
        -----
        Empty rows are skipped. SMILES validation and mol_block generation (if empty)
        are handled automatically by the Ligand model validator.
        """
        if name_col not in df.columns:
            raise KeyError(f"Column '{name_col}' not found in DataFrame")
        if smi_col not in df.columns:
            raise KeyError(f"Column '{smi_col}' not found in DataFrame")
        
        items = []
        for _, row in df.iterrows():
            name_val = row[name_col]
            smiles_val = row[smi_col]
            
            # Handle NaN values (empty strings in CSV become NaN when read)
            if pd.isna(name_val) or pd.isna(smiles_val):
                continue
            
            name = str(name_val).strip()
            smiles = str(smiles_val).strip()
            
            # Skip rows with empty name or SMILES (validator will catch invalid SMILES)
            if not name or not smiles:
                continue
            
            # Let validator handle SMILES validation and mol_block generation
            items.append({
                "name": name,
                "smiles": smiles,
                "mol_block": "",
                "source": source
            })
        
        return items

    def _validate_uniqueness(self, ligands: List[Ligand]):
        """
        Validate that all ligand names are unique.

        Parameters
        ----------
        ligands : List[Ligand]
            List of ligand objects to validate.

        Raises
        ------
        ValueError
            If duplicate names are found.

        Notes
        -----
        Also validates that names are non-empty (enforced by Ligand model).
        """
        names = [l.name for l in ligands]
        if len(names) != len(set(names)):
            dupes = [k for k, v in Counter(names).items() if v > 1]
            raise ValueError(f"Duplicate ligand names detected: {dupes}")
