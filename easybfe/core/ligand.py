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
import warnings

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
            self.mol_block = Chem.MolToMolBlock(m)
        else:
            mol_from_block = Chem.MolFromMolBlock(self.mol_block)
            smi_from_block = Chem.MolToSmiles(Chem.RemoveHs(mol_from_block))
            assert smi_from_block == self.smiles, 'SMILES and mol block does not refer to the same molecule'
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
    def from_directory(cls, directory: os.PathLike, stem: Optional[str] = None):
        directory = Path(directory).expanduser().resolve()
        stem = Path(directory).stem if stem is not None else stem
        loader = LigandLoader()
        sdf = directory / f'{stem}.sdf'
        if not sdf.is_file():
            sdfs = list(directory.glob("*.sdf"))
            if len(sdfs) > 1:
                raise RuntimeError(f"Multiple sdf files found in {directory}")
            elif len(sdfs) == 0:
                raise FileNotFoundError(f"No sdf file found in {directory}")
            else:
                sdf = sdfs[0]
                stem = sdf.stem
        ligand = loader.load(directory / f'{stem}.sdf', only_first=True, name_from_stem=True)[0]
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
    
    def to_openmm(self):
        import openmm.app as app
        _io = io.StringIO(self.auxiliary_files['pdb'])
        _io.seek(0)
        return app.PDBFile(_io)
          

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

    def load(
        self,
        source: Any,
        enforce_unique_name: bool = True,
        name_from_stem: bool = True,
        only_first: bool = False,
        name_prop: str = "_Name",
        name_col: Optional[str] = None,
        smi_col: str = "smiles",
        smi_col_index: int = 0,
        auto_naming: bool = False,
        name: Optional[str] = None
    ) -> List[Ligand]:
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
        name_from_stem : bool, optional
            Use filename stem as ligand name for SDF/SMI files. Default True.
        only_first : bool, optional
            Only read first molecule from each SDF/SMI file. Default False.
        name_prop : str, optional
            RDKit property used for molecule name in SDF. Default ``'_Name'``.
        name_col : str, optional
            Column name for ligand names (DataFrame/CSV). Required for DataFrame and CSV.
        smi_col : str, optional
            Column name for SMILES (DataFrame/CSV). Required for DataFrame and CSV.
        smi_col_index : int, optional
            Column index for SMILES in SMI files (0 or 1). Default 0.
        auto_naming : bool, optional
            Assign default names when name is missing. Default False.

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
        if isinstance(source, pd.DataFrame):
            items = self._from_dataframe_or_csv(
                source,
                smi_col=smi_col,
                name_col=name_col,
                source="DataFrame",
                auto_naming=auto_naming,
            )
        elif isinstance(source, list) and len(source) > 0 and isinstance(source[0], Chem.Mol):
            items = self._from_rdkit_mol_list(source)
        elif isinstance(source, (str, Path, list)):
            items = self._from_files(
                source,
                name_from_stem=name_from_stem,
                only_first=only_first,
                name_prop=name_prop,
                name_col=name_col,
                smi_col=smi_col,
                smi_col_index=smi_col_index,
                auto_naming=auto_naming,
            )
        else:
            raise ValueError(f"Unsupported input type: {type(source)}")

        ligands = [Ligand(**item) for item in items]
        if enforce_unique_name:
            self._validate_uniqueness(ligands)
        
        if name:
            if len(ligands) == 0:
                ligands[0].name = name
            else:
                warnings.warn(f"Multiple ligands loaded - name='{name}' will not be effective.")
        return ligands

    # --- Internal parsing logic (private methods) ---

    def _from_files(
        self,
        paths: Union[str, Path, list[Union[str, Path]]],
        name_from_stem: bool = True,
        only_first: bool = False,
        name_prop: str = "_Name",
        name_col: Optional[str] = None,
        smi_col: Optional[str] = None,
        smi_col_index: int = 0,
        auto_naming: bool = False
    ) -> list[dict]:
        """
        Parse ligands from file paths.

        Parameters
        ----------
        paths : str, Path, or List[Union[str, Path]]
            Single file path or list of file paths.
        name_from_stem : bool, optional
            If True, use the filename stem as the ligand name for SDF/SMI inputs
            whenever there is exactly one molecule per file. For multi-molecule
            SDF files, the ``_Name`` property is used instead and a warning is
            emitted. Default True.
        only_first : bool, optional
            Only read first molecule from each SDF/SMI file. Default True.
        name_prop : str, optional
            RDKit property used for molecule name in SDF. Default ``'_Name'``.
        name_col : str, optional
            Column name for ligand names in CSV. Required for CSV files.
        smi_col : str, optional
            Column name for SMILES in CSV. Required for CSV files.
        smi_col_index : int, optional
            Column index for SMILES in SMI files (0 or 1). Default 0.
        auto_naming : bool, optional
            If True, assign default names (e.g. ``{filename}_mol_{i}``) when
            name is missing. Default True.

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
        single_path = isinstance(paths, (str, Path))
        path_list = [Path(paths)] if single_path else [Path(p) for p in paths]
        # Honor name_from_stem for both single and multiple paths; the per-file
        # logic in parsers will fall back to molecule properties when needed.
        use_stem = name_from_stem
        all_items = []

        for p in path_list:
            if not p.is_file():
                raise FileNotFoundError(f"File not found: {p}")

            if p.suffix.lower() == ".sdf":
                all_items.extend(
                    self._parse_sdf(
                        p, use_stem, only_first,
                        name_prop=name_prop, auto_naming=auto_naming
                    )
                )
            elif p.suffix.lower() == ".csv":
                if name_col is None or smi_col is None:
                    raise ValueError(
                        "name_col and smi_col must be provided for CSV files"
                    )
                all_items.extend(
                    self._from_dataframe_or_csv(
                        p,
                        smi_col=smi_col,
                        name_col=name_col,
                        source=p.name,
                        auto_naming=auto_naming,
                    )
                )
            elif p.suffix.lower() in [".smi", ".smiles"]:
                all_items.extend(
                    self._parse_smi(
                        p, use_stem, only_first, smi_col_index,
                        auto_naming=auto_naming
                    )
                )
            else:
                raise ValueError(f"Unsupported file format: {p.suffix}")

        return all_items

    def _parse_sdf(
        self, fpath: Path, use_stem: bool, only_first: bool, name_prop: str = '_Name', auto_naming: bool = False
    ) -> List[dict]:
        """
        Parse ligands from an SDF file with 3D geometry.

        Parameters
        ----------
        fpath : Path
            Path to the SDF file.
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
        suppl = Chem.SDMolSupplier(str(fpath), removeHs=False)
        mols = [next(suppl)] if only_first else [m for m in suppl]
        for i, m in enumerate(mols):
            if m is None:
                raise ValueError(f"Fail to parse molecule {i} in {fpath}")
        if len(mols) > 1:
            if use_stem:
                warnings.warn(f"Muliple ligands found in {fpath}. use_stem will be forcibly to be False. Molecule names will be read from {name_prop}")
            names = []
        else:
            names = [fpath.stem for _ in mols] if use_stem else []

        items = self._from_rdkit_mol_list(mols, name_prop=name_prop, names=names)
        for i in range(len(items)):
            items[i]["source"] = fpath.name
            if auto_naming and not items[i]['name']:
                items[i]['name'] = f'{fpath.name}_mol_{i}'
        return items

    def _parse_smi(self, fpath: Path, use_stem: bool, only_first: bool, smi_col_index: int = 0, auto_naming: bool = False) -> List[dict]:
        """
        Parse ligands from a SMILES file.

        Parameters
        ----------
        fpath : Path
            Path to the SMILES file.

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
        text = fpath.read_text(encoding="utf-8")
        items = []

        sc = 0 if smi_col_index == 0 else 1
        nc = 1 if smi_col_index == 0 else 0

        for line_num, line in enumerate(text.strip().split("\n"), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            smiles = parts[sc]
            name = parts[nc].strip() if len(parts) > 1 else ""

            items.append({
                "name": name,
                "smiles": smiles,
                "mol_block": "",
                "source": fpath.name
            })

            if only_first:
                break

        if use_stem and items:
            if len(items) > 1:
                warnings.warn(
                    f"Multiple ligands found in {fpath}. "
                    "use_stem will be forcibly set to False."
                )
            else:
                items[0]["name"] = fpath.stem

        if auto_naming:
            for i in range(len(items)):
                if not items[i]["name"]:
                    items[i]["name"] = f"{fpath.name}_mol_{i}"
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

    def _from_rdkit_mol_list(self, mols: List[Chem.Mol], name_prop: str = '_Name', names: list[str] = list(), auto_naming: bool = True) -> List[dict]:
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
        if auto_naming:
            for i in range(len(items)):
                if not items[i]["name"]:
                    items[i]["name"] = f"mol_{i}"
        return items

    def _from_dataframe_or_csv(
        self, 
        data: Union[pd.DataFrame, str, Path], 
        smi_col: str = 'smiles', 
        name_col: Optional[str] = None,
        source: str = "DataFrame",
        auto_naming: bool = False
    ) -> List[dict]:
        """
        Parse ligands from a pandas DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
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
        if not isinstance(data, pd.DataFrame):
            path = Path(data)
            df = pd.read_csv(str(path))
            return self._from_dataframe_or_csv(
                df,
                smi_col=smi_col,
                name_col=name_col,
                source=path.name,
                auto_naming=auto_naming,
            )

        if (name_col) and name_col not in data.columns:
            raise KeyError(f"Column '{name_col}' not found in DataFrame")
        if smi_col not in data.columns:
            raise KeyError(f"Column '{smi_col}' not found in DataFrame")
        
        items = []
        for _, row in data.iterrows():
            name_val = row[name_col] if name_col else ""
            smiles_val = row[smi_col]
            
            # Handle NaN values (empty strings in CSV become NaN when read)
            if pd.isna(smiles_val):
                continue
            
            name = str(name_val).strip()
            smiles = str(smiles_val).strip()

            if not smiles.strip():
                continue
            
            # Let validator handle SMILES validation and mol_block generation
            items.append({
                "name": name,
                "smiles": smiles,
                "mol_block": "",
                "source": source
            })
        
        if auto_naming:
            for i in range(len(items)):
                if not items[i]['name']:
                    items[i]['name'] = f'{source}_mol_{i}'
        
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
