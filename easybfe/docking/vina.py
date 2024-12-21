import os
from pathlib import Path
from typing import Dict, List
import tempfile
import shutil
from copy import deepcopy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTMolecule, RDKitMolCreate

from .base import compute_box_from_coordinates
from ..cmd import run_command, find_executable, init_logger
from .embed import constrained_optimization
from ..mapping import OpenFEAtomMapper, LazyMCSMapper


def convert_pdb_to_pdbqt(protein_pdb: os.PathLike, output_path: os.PathLike, tool: str = 'adfr'):
    if tool == 'adfr':
        exec = find_executable('prepare_receptor')
        run_command([exec, '-r', protein_pdb, '-o', output_path, '-A', 'checkhydrogens'])
    elif tool == 'obabel':
        exec = find_executable('obabel')
        run_command([exec, '-ipdb', protein_pdb, '-opdbqt', '-O', output_path, '-xr'])
    else:
        raise NotImplementedError(f'Unsupported tool: {tool}')


def convert_mol_to_pdbqt(mol, pdbqt_path):
    prep = MoleculePreparation()
    prep.prepare(mol)
    prep.write_pdbqt_file(str(pdbqt_path))


def convert_sdf_to_pdbqt(sdf_path, pdbqt_path):
    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    convert_mol_to_pdbqt(mol, pdbqt_path)


def convert_pdbqt_to_sdf(pdbqt_path, sdf_path):
    pdbqt_mol = PDBQTMolecule.from_file(str(pdbqt_path), skip_typing=True)
    sdstring, _ = RDKitMolCreate.write_sd_string(pdbqt_mol)
    with open(sdf_path, 'w') as f:
        f.write(sdstring)


def write_config(config: Dict, fpath: os.PathLike):
    with open(fpath, 'w') as f:
        for k, v in config.items():
            f.write(f'{k} = {v}\n')


def compute_rmsd(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2) ** 2) / pos1.shape[0])


class VinaDocking:
    
    def __init__(
        self, 
        protein: os.PathLike, 
        box_center: List[float] = [0.0, 0.0, 0.0], 
        box_size: List[float] = [0.0, 0.0, 0.0], 
        wdir: os.PathLike = '.', 
        **kwargs
    ):
        self.vina_binary = kwargs.pop('vina_binary', find_executable('vina'))
        self.wdir = Path(wdir).resolve()
        self.wdir.mkdir(exist_ok=True)
        
        protein = Path(protein).resolve()
        output_path = self.wdir / f'{protein.stem}.pdbqt'
        if protein.suffix == '.pdb':
            tool = kwargs.pop('protein_prep_tool', 'adfr')
            convert_pdb_to_pdbqt(protein, output_path, tool)
        elif protein.suffix == '.pdbqt':
            shutil.copyfile(protein, output_path)
        else:
            raise RuntimeError(f"Unsupport protein format: {protein.suffix}")
        self.protein = protein
        self.protein_pdbqt = output_path
            
        self.config = {
            'receptor': output_path,
            'center_x': box_center[0],
            'center_y': box_center[1],
            'center_z': box_center[2],
            'size_x': box_size[0], 
            'size_y': box_size[1], 
            'size_z': box_size[2],
            "num_modes": 20,
            "min_rmsd": 0.5,
            "energy_range": 5,
            "exhaustiveness": 8
        }
        self.config.update(kwargs)
        self.logger = init_logger()
    
    def update_box(self, center, box):
        self.config['center_x'] = center[0]
        self.config['center_y'] = center[1]
        self.config['center_z'] = center[2]
        self.config['size_x'] = box[0]
        self.config['size_y'] = box[1]
        self.config['size_z'] = box[2]

    def write_config(self):
        self.config_file = self.wdir / 'config.txt'
        write_config(self.config, self.config_file)

    def dock(self, in_sdf: os.PathLike, output_dir: os.PathLike | None = None):
        self.write_config()
        stem = Path(in_sdf).stem
        
        if output_dir is None:
            dock_dir = self.wdir / stem
        else:
            dock_dir = Path(output_dir)
        
        dock_dir.mkdir(exist_ok=True)

        in_pdbqt = dock_dir / f'{stem}.pdbqt'
        out_pdbqt = dock_dir / f'{stem}_out.pdbqt'
        out_sdf = dock_dir / f'{stem}_out.sdf'
        convert_sdf_to_pdbqt(in_sdf, in_pdbqt)
        run_command([self.vina_binary, '--ligand', in_pdbqt, '--out', out_pdbqt, '--config', self.config_file], logger=self.logger)
        convert_pdbqt_to_sdf(out_pdbqt, out_sdf)
        self.logger.info(f"Results are written to: {out_sdf}")
        return out_sdf
    
    def rescore(self, inp: os.PathLike | Chem.Mol):
        in_pdbqt = tempfile.mkstemp(suffix='.pdbqt')[1]
        if isinstance(inp, Chem.Mol):
            convert_mol_to_pdbqt(inp, in_pdbqt)
        else:
            convert_sdf_to_pdbqt(inp, in_pdbqt)
        cmd = [
            self.vina_binary, '--ligand', in_pdbqt, '--receptor', self.protein_pdbqt, '--score_only', '--autobox'
        ]
        return_code, out, err = run_command(cmd)
        # parse output
        lines = out.split('\n')
        energies = {'binding': 0.0, 'inter': 0.0, 'torsion': 0.0}
        for line in lines:
            if line.startswith('Estimated Free Energy of Binding'):
                energies['binding'] = float(line.split()[-3])
            elif line.startswith('(1) Final Intermolecular Energy'):
                energies['inter'] = float(line.split()[-2])
            elif line.startswith('(3) Torsional Free Energy'):
                energies['torsion'] = float(line.split()[-2])
        return energies
    
    def constr_dock(self, in_smi_or_sdf: str | os.PathLike, ref_sdf: os.PathLike, name: str = "", thresh: float = 5.0):
        # set up name and result directory
        if os.path.isfile(in_smi_or_sdf):
            mol = Chem.SDMolSupplier(in_smi_or_sdf, removeHs=False)[0]
            name = Path(in_smi_or_sdf).stem if not name else name
        else:
            mol = Chem.AddHs(Chem.MolFromSmiles(in_smi_or_sdf))
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            assert name, "Must provide a name"
        
        output_dir = self.wdir / name
        output_dir.mkdir(exist_ok=True)

        # Update box according to reference sdf
        ref = Chem.SDMolSupplier(ref_sdf, removeHs=False)[0]
        center, box = compute_box_from_coordinates(ref.GetConformer().GetPositions())
        self.update_box(center, box)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_sdf = os.path.join(tmpdir, f'{name}.sdf')
            writer = Chem.SDWriter(in_sdf)
            writer.write(mol)
            writer.close()
            out_sdf = self.dock(in_sdf)
        
        mols = [m for m in Chem.SDMolSupplier(str(out_sdf), removeHs=False)]

        mapping = OpenFEAtomMapper('lomap', element_change=False, max3d=10.0).run(mols[0], ref, output_dir)

        mapping_heavy_only = {}
        for k, v in mapping.items():
            if mols[0].GetAtomWithIdx(k).GetSymbol() == 'H' or ref.GetAtomWithIdx(v).GetSymbol() == 'H':
                continue
            mapping_heavy_only[k] = v
        
        prb_indices, ref_indices = [], []
        for k, v in mapping_heavy_only.items():
            prb_indices.append(k)
            ref_indices.append(v)

        accepted = []
        for mol in mols:
            conf = mol.GetConformer()
            # compute rmsd
            rmsd = compute_rmsd(
                conf.GetPositions()[prb_indices],
                ref.GetConformer().GetPositions()[ref_indices]
            )
            # rejected if rmsd > thresh
            if rmsd > thresh:
                continue
            accepted.append(mol)
            mol.SetDoubleProp("rmsd", rmsd)
        
        assert len(accepted) > 0, f"Constrained docking fails because no poses with RMSD < {thresh:.2f} angstrom w.r.t reference"
        tmp_sdf = output_dir / f'{name}_selected.sdf'
        writer = Chem.SDWriter(tmp_sdf)
        for i, mol in enumerate(accepted):
            writer.write(mol)
        writer.close()
        self.logger.info(f"Found {len(accepted)} docked pose with RMSD < {thresh} angstrom w.r.t reference. Results in {tmp_sdf}")
        
        # Gather all conformers to a single molecule object
        out_mol = deepcopy(accepted[0])
        for m in accepted[1:]:
            out_mol.AddConformer(m.GetConformer(), assignId=True)

        self.logger.info("Running optimization")
        coord_map = {k: ref.GetConformer().GetPositions()[v] for k, v in mapping.items()}
        assert self.protein.suffix == '.pdb', 'Input protein must be a .pdb file'
        out_mol, energies = constrained_optimization(out_mol, self.protein, coord_map, constr=True, restr=10)
        for i, m in enumerate(accepted):
            m.SetProp("_Name", name)
            m.ClearComputedProps()
            m.SetDoubleProp("ff.energy", float(energies[i]))
            m.GetConformer().SetPositions(out_mol.GetConformer(i).GetPositions())
            rescore = self.rescore(m)
            m.SetDoubleProp('vina.score', rescore['binding'])
            m.SetProp('vina.score.details', str(rescore))
        accepted.sort(key=lambda m: m.GetDoubleProp('vina.score'))

        for i, mol in enumerate(accepted):
            out_sdf = output_dir / f'{name}_constr_dock_{i}.sdf'
            writer = Chem.SDWriter(out_sdf)
            writer.write(mol)
            writer.close()
        self.logger.info(f"Results written to: {output_dir}")