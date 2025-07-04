import os, glob
from pathlib import Path
import json
from typing import Optional, Dict, Any, Tuple, List
from functools import partial
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import pandas as pd

from ...cmd import run_command, set_directory

from .base import BaseAmberRbfeProject, SimulationStatus


class AmberLigandRbfeProject(BaseAmberRbfeProject):
    """
    Project to run ligand RBFE
    """
    @property
    def perturbations(self) -> Dict[str, List[str]]:
        results = defaultdict(list)
        for pert_dir in self.rbfe_dir.glob('*/*'):
            if not Path.is_dir(pert_dir) or pert_dir.name.startswith('.'):
                continue
            results[pert_dir.parent.name].append(pert_dir.name)
        return results
    
    def add_perturbation(
        self, 
        ligandA_name: str, 
        ligandB_name: str, 
        protein_name: str,
        pert_name: Optional[str] = None, 
        config: Dict[str, Any] | os.PathLike = dict(),
        submit: bool = False,
        skip_gas: bool = True,
        overwrite: bool = False
    ):
        """
        Create a perturbation pair (relative binding free energy between ligand A and ligand B) 
        and set up FEP simulation

        Parameters
        ----------
        ligandA_name: str
            Name of ligand A
        ligandB_name: str
            Name of ligand B
        protein_name: str
            Name of the protein
        pert_name: str
            Name of this perturbation. If None, will use `ligandA_name`~`ligandB_name`.`protein_name`
        mcs: str or os.PathLike
            Maximum common structure (MCS) between ligand A and B. If a sdf file or a SMARTS string is provided, rdkit will be used 
            to parse it. If None, will use rdkit to determine the MCS. Manually provided MCS is highly recommended.
        config: dict or os.PathLike
            JSON-formatted configuration file or dictionary. Must be provided.
        submit: bool
            Whether to use SLURM (sbatch command) to submit the simulation job. Default False.
        skip_gas: bool
            Whether to skip the gas-phase FEP simulation, only useful when `submit=True`. Default True.
        overwrite: bool
            Whether to overwrite existing perturbation directory. Default False.
        patch_func: Callable
            Function to modify the regular behavior of this function, used for modifying force field parameters
        """
        from rdkit import Chem
        from ...mapping import LazyMCSMapper, OpenFEAtomMapper
        from ..prep import prep_ligand_rbfe_systems
        from ..op import fep_workflow

        # Read config from json file
        if not isinstance(config, dict):
            with open(config) as f:
                config = json.load(f)
        
        # Setup
        if pert_name is None:
            pert_name = f"{ligandA_name}~{ligandB_name}"
        
        pert_dir = self.rbfe_dir / protein_name / pert_name
        if (not overwrite) and pert_dir.is_dir():
            msg = f'Perturbation already exists: {pert_dir}'
            self.logger.error(msg)
            raise RuntimeError(msg)
            
        pert_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f'Creating dirctory: {pert_dir}')
        
        # Read Ligands
        ligandA = Chem.SDMolSupplier(str(self.ligands_dir / protein_name / ligandA_name / f'{ligandA_name}.sdf'), removeHs=False)[0]
        ligandB = Chem.SDMolSupplier(str(self.ligands_dir / protein_name / ligandB_name / f'{ligandB_name}.sdf'), removeHs=False)[0]

        # Read binding affinity
        with open(self.ligands_dir / protein_name / ligandA_name / 'info.json') as f:
            dG_ligA = json.load(f).get('dG.expt', None)
        with open(self.ligands_dir / protein_name / ligandB_name / 'info.json') as f:
            dG_ligB = json.load(f).get('dG.expt', None)
        
        if dG_ligA is not None and dG_ligB is not None:
            ddG_expt = dG_ligB - dG_ligA
        else:
            ddG_expt = None
        
        # Write basic info
        with open(pert_dir / 'info.json', 'w') as f:
            basic_info = {
                "protein_name": protein_name,
                "ligandA_name": ligandA_name, 
                "ligandB_name": ligandB_name,
                "ddG.expt": ddG_expt,
            }
            json.dump(basic_info, f, indent=4)

        # Atom Mapping
        atom_mapping_method = config.get('atom_mapping_method', "kartograf")
        atom_mapping_options = config.get('atom_mapping_options', dict())
        self.logger.info(f"Generate atom mapping with {atom_mapping_method}")
        if atom_mapping_options:
            self.logger.info(f"The following customized options are used for atom mapping: {atom_mapping_options}")
        
        if atom_mapping_method == 'lazymcs':
            mcs = atom_mapping_options.get('mcs', None)
            if mcs is None:
                mcs_mol = None
                self.logger.info("No MCS passed in. Will use RDKit to generate one.")
                pass
            elif isinstance(mcs, Chem.Mol):
                mcs_mol = mcs
            elif isinstance(mcs, Path) or (isinstance(mcs, str) and os.path.isfile(mcs)):
                self.logger.info(f"Read MCS from {mcs}")
                mcs_mol = Chem.SDMolSupplier(str(mcs), removeHs=True)[0]
            else:
                mcs_mol = Chem.MolFromSmiles(mcs)
                if mcs_mol is None:
                    self.logger.info("MCS is not a valid SMILES. Parsing MCS as a SMARTS.")
                    mcs_mol = Chem.MolFromSmarts(mcs)
                if mcs_mol is None:
                    msg = f"Unrecognized MCS: '{mcs}'. Maybe this is not a valid SMILES or SMARTS or file path"
                    self.logger.error(msg)
                    raise RuntimeError(msg)
            atom_mapping_options['mcs'] = mcs_mol
            mapper = LazyMCSMapper(**atom_mapping_options)
        else:
            mapper = OpenFEAtomMapper(method=atom_mapping_method, **atom_mapping_options)    
        atom_mapping = mapper.run(ligandA, ligandB, pert_dir)

        tol = 0.1
        for k, v in atom_mapping.items():
            posA = ligandA.GetConformer().GetAtomPosition(k)
            posB = ligandB.GetConformer().GetAtomPosition(v)
            dist = np.linalg.norm(posA - posB)
            if dist >= tol:
                msg = f"Distance ({dist:.2f} angstrom) between mapped atoms Atom(#{k}) in ligand A and Atom(#{v}) in ligand B larger than {tol:.2f} angstrom"
                self.logger.warning(msg)
                ligandB.GetConformer().SetAtomPosition(v, [float(x) for x in posA])
                self.logger.warning(f"Reset Atom(#{v}) position: {(posB.x, posB.y, posB.z)} -> {(posA.x, posA.y, posA.z)}")

        # Prep simulation system
        self.logger.info("Preparing simulation system")
        prep_dir = pert_dir / 'prep'
        prep_dir.mkdir(exist_ok=True)
        charge_change_mdin_mod = prep_ligand_rbfe_systems(
            self.proteins_dir / protein_name / f'{protein_name}.pdb',
            ligandA,
            self.ligands_dir / protein_name / ligandA_name / f'{ligandA_name}.prmtop',
            ligandB,
            self.ligands_dir / protein_name / ligandB_name / f'{ligandB_name}.prmtop',
            mapping=atom_mapping,
            wdir=prep_dir,
            protein_ff=config.get('protein_ff', 'ff14SB'),
            water_ff=config.get('water_ff', 'tip3p'),
            gas_config=config.get('gas', {}),
            solvent_config=config.get('solvent', {}),
            complex_config=config.get('complex', {}),
            use_charge_change=config.get('use_charge_change', True),
            use_settle_for_alchemical_water=config.get('use_settle_for_alchemical_water', True)
        )

        # Prep workflow
        self.logger.info("Preparing Amber simulation inputs")
        for leg in ['gas', 'solvent', 'complex']:
            leg_dir = pert_dir / leg
            leg_dir.mkdir(exist_ok=True)
            prmtop = pert_dir / f'prep/{leg}.prmtop'
            inpcrd = pert_dir / f'prep/{leg}.inpcrd'

            leg_config = {
                "inpcrd": str(inpcrd),
                "prmtop": str(prmtop),
            }
            leg_config.update(config[leg])

            with open(leg_dir / 'config.json', 'w') as f:
                json.dump(leg_config, f, indent=4)

            fep_workflow(
                leg_config, leg_dir, 
                gas_phase=(leg == 'gas'), 
                charge_change_mdin_mod=charge_change_mdin_mod.get(leg, [])
            )
            self.logger.info(f"FEP simulation workflow is set for leg: {leg}. Config file written to: {leg_dir / 'config.json'}")

            with open(leg_dir / 'run.slurm', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('\n'.join(config.get('header', [])))
                f.write('\n')
                f.write('source run.sh')
        
        # submit
        if submit:
            legs = ['solvent', 'complex'] if skip_gas else ['solvent', 'complex', 'gas']
            for leg in legs:
                with set_directory(pert_dir / leg):
                    _, out, _ = run_command(['sbatch', 'run.slurm'])
                self.logger.info(f"Job submitted for {leg}: {out.split()[-1]}")
    
    def add_perturbations(
        self,
        perts: os.PathLike | List[Tuple[str, str]],
        protein_name: str,
        num_workers: str | int = 'auto',
        **kwargs
    ):
        """
        Add Perturbations
        """
        if num_workers == 'auto':
            num_workers = max(1, mp.cpu_count() - 2)

        if isinstance(perts, str) or isinstance(perts, Path):
            with open(perts) as f:
                perts = [line.split() for line in f.read().strip().split('\n')]
        if num_workers == 1:
            for pert in perts:
                self.add_perturbation(pert[0], pert[1], protein_name, **kwargs)
        else:
            func = partial(self.add_perturbation, protein_name=protein_name, **kwargs)
            with mp.Pool(num_workers) as pool:
                pool.starmap(func, perts)

    def analyze_pert(self, protein_name: str, pert_name: str, skip_traj: bool = False):
        self.evaluate_free_energy(protein_name, pert_name)
        if not skip_traj:
            self.process_traj(protein_name, pert_name)
    
    @classmethod
    def query_perturbation_status(cls, pert_dir: Path):
        status = {
            'analysis': Path.is_file(pert_dir / 'result.json')
        }
        legs = ['gas', 'solvent', 'complex']
        for leg in legs:
            if Path.is_file(pert_dir / leg / 'done.tag'):
                status[leg] = SimulationStatus.FINISHED
            elif Path.is_file(pert_dir / leg / 'error.tag'):
                status[leg] = SimulationStatus.ERROR
            elif Path.is_file(pert_dir / leg / 'running.tag'):
                status[leg]= SimulationStatus.RUNNING
            else:
                status[leg] = SimulationStatus.NOTSTART
        return status
    
    @classmethod
    def query_perturbation_info(cls, pert_dir: Path):
        with open(pert_dir / 'info.json') as f:
            info = json.load(f)
        info['pert_name'] = pert_dir.name
        status = cls.query_perturbation_status(pert_dir)
        info['analysis.finished'] = status['analysis']
        for leg in ['gas', 'solvent', 'complex']:
            info[f'{leg}.status'] = status[leg].name
        if info['analysis.finished']:
            with open(pert_dir / 'result.json') as f:
                result = json.load(f)
            for key1 in result:
                for key2 in result[key1]:
                    info[f'{key1}.{key2}'] = result[key1][key2]
        return info
    
    def gather_perturbations_info(self):
        '''Gather data of all perturbations'''

        infos = []
        for protein_name, pert_names in self.perturbations.items():
            for pert_name in pert_names:
                info = self.query_perturbation_info(self.rbfe_dir / protein_name / pert_name)
                infos.append(info)

                with open(self.ligands_dir / info['protein_name'] / info['ligandA_name'] / 'info.json') as f:
                    dG_ligA = json.load(f)['dG.expt']
                with open(self.ligands_dir / info['protein_name'] / info['ligandB_name'] / 'info.json') as f:
                    dG_ligB = json.load(f)['dG.expt']
                if dG_ligA is not None and dG_ligB is not None:
                    info['ddG.expt'] = dG_ligB - dG_ligA
                else:
                    info['ddG.expt'] = None

                with open(self.rbfe_dir / protein_name / pert_name / 'info.json', 'w') as f:
                    json.dump(info, f, indent=4)

        df = pd.DataFrame(infos)
        # empty df
        if df.shape[0] == 0:
            return df
        df = df.sort_values(by=['protein_name', 'ligandA_name', 'ligandB_name'])
        return df
    
    def analyze_pert(self, protein_name: str, pert_name: str, skip_traj: bool = False):
        self.evaluate_free_energy(protein_name, pert_name)
        if not skip_traj:
            self.process_traj(protein_name, pert_name)
    
    def analyze(self, num_workers: int = 1, skip_traj: bool = False):
        info_df = self.gather_perturbations_info()
        query_str = '~`analysis.finished` & `solvent.status` == "{}" & `complex.status` == "{}"'.format(
            SimulationStatus.FINISHED.name, SimulationStatus.FINISHED.name
        )
        ana_df = info_df.query(query_str)
        print_df = ana_df[['protein_name', 'ligandA_name', 'ligandB_name', 'ddG.expt']]
        self.logger.info(f"Found these perturbations needs analysis:\n{ana_df}")

        args = [(row['protein_name'], row['pert_name'], skip_traj) for _, row in ana_df.iterrows()]
        with mp.Pool(num_workers) as pool:
            pool.starmap(self.analyze_pert, args)
    
    def report(self, save_dir: os.PathLike, protein_name: str = '', verbose: bool = False):
        """Report data"""
        from ...analysis import maximum_likelihood_estimator, plot_correlation

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        ligands_info = self.gather_ligands_info()
        perts_info = self.gather_perturbations_info()

        if not protein_name:
            protein_name = self.proteins[0]
            self.logger.info(f"No protein name is provided. Will use the first protein in the project: {protein_name}")
        ligands_info = ligands_info.query(f'`protein` == "{protein_name}"')
        perts_info = perts_info.query(f'`protein_name` == "{protein_name}"')
        
        ligands_with_expt = ligands_info.dropna(subset=['dG.expt'])['name'].tolist()
        dg_mle = maximum_likelihood_estimator(perts_info.dropna(subset=['ddG.total'])).set_index('ligand')
        avg_dg_expt = ligands_info.query('name in @ligands_with_expt')['dG.expt'].mean()
        avg_dg_calc = dg_mle[dg_mle.index.isin(ligands_with_expt)]['dG'].mean()
        dg_mle['dG'] += avg_dg_expt - avg_dg_calc
        for index, row in ligands_info.iterrows():
            if row['name'] in dg_mle.index:
                dG = dg_mle.loc[row['name'], 'dG']
                dG_std = dg_mle.loc[row['name'], 'dG_std']
            else:
                dG, dG_std = None, None
            ligands_info.loc[index, 'dG.calc'] = dG
            ligands_info.loc[index, 'dG_std.calc'] = dG_std
        ligands_info['error'] = np.abs(ligands_info['dG.calc'] - ligands_info['dG.expt'])
        ligands_info = ligands_info.sort_values('error')

        perts_info['error'] = np.abs(perts_info['ddG.expt'] - perts_info['ddG.total'])
        perts_info = perts_info.sort_values('error')
        if not verbose:
            perts_info = perts_info[['protein_name', 'ligandA_name', 'ligandB_name', 'ddG.expt', 'ddG.total', 'ddG_std.total', 'error']]
        
        ligands_info.to_csv(os.path.join(save_dir, 'ligands.csv'), index=None)
        perts_info.to_csv(os.path.join(save_dir, 'perturbations.csv'), index=None)
        
        ligands_info_with_expt = ligands_info.dropna(subset=['dG.expt', 'dG.calc'])
        self.logger.info(f"Plotting - Found {ligands_info_with_expt.shape[0]} ligands with both experimental values and calculated values")
        _, ligands_stats = plot_correlation(
            xdata=ligands_info_with_expt['dG.expt'],
            ydata=ligands_info_with_expt['dG.calc'],
            xerr=None,
            yerr=ligands_info_with_expt['dG_std.calc'],
            xlabel=r'$\Delta G_\mathrm{expt}$ (kcal/mol)',
            ylabel=r'$\Delta G_\mathrm{FEP}$ (kcal/mol)',
            savefig=os.path.join(save_dir, 'ligands.png')
        )
        with open(os.path.join(save_dir, 'ligands_stat.json'), 'w') as f:
            json.dump(ligands_stats, f, indent=4)

        perts_info_with_expt = perts_info.dropna(subset=['ddG.expt', 'ddG.total'])
        self.logger.info(f"Plotting - Found {perts_info_with_expt.shape[0]} perturbations with both experimental values and calculated values")
        _, perts_stats = plot_correlation(
            xdata=perts_info_with_expt['ddG.expt'],
            ydata=perts_info_with_expt['ddG.total'],
            xerr=None,
            yerr=perts_info_with_expt['ddG_std.total'],
            xlabel=r'$\Delta\Delta G_\mathrm{expt}$ (kcal/mol)',
            ylabel=r'$\Delta\Delta G_\mathrm{FEP}$ (kcal/mol)',
            savefig=os.path.join(save_dir, 'perturbations.png')
        )
        with open(os.path.join(save_dir, 'perturbations_stat.json'), 'w') as f:
            json.dump(perts_stats, f, indent=4)

    def evaluate_free_energy(self, protein_name: str, pert_name: str):
        """
        Analyze FEP simulation results using `alchemlyb` package.
        Free energy will be estimated using MBAR, overlap matrix and convergence analysis are also performed.

        For details, one can refer to: J Comput Aided Mol Des (2015) 29:397-411

        Parameters
        ----------
        pert_name: str
            Name of the perturbation to analyze
        skip_gas: bool
            Whether to skip gas-phase simulation analysis. Default is True.
        """
        from alchemlyb.visualisation.convergence import plot_convergence
        from ...analysis.mbar import run_mbar

        dG = {}
        dG_std = {}

        ddG = {}
        ddG_std = {}

        pert_dir = self.rbfe_dir / protein_name / pert_name

        skip_gas = not os.path.isfile(pert_dir / 'gas' / 'done.tag')
        self.logger.info("Found gas-pahse simulation")

        legs = ['solvent', 'complex'] if skip_gas else ['solvent', 'complex', 'gas']
        for leg in legs:
            leg_dir = pert_dir / leg
            with open(leg_dir / 'config.json') as f:
                T = json.load(f).get('temperature', 298.15)

            self.logger.info(f"Performing MBAR for {leg}")
            self.logger.info("Extracting data from output...")
            dg, dg_std = run_mbar(leg_dir, T, self.logger)
            dG[leg] = dg
            dG_std[leg] = dg_std

        convergence = {leg: pd.read_csv(str(pert_dir / leg / 'convergence.csv')) for leg in legs}

        pairs = [('complex', 'solvent'), ('solvent', 'gas'), ('complex', 'gas')]
        names = ['total', 'solvation', 'complex']

        for (leg1, leg2), name in zip(pairs, names):
            ddG[name] = dG[leg1] - dG[leg2]
            ddG_std[name] = np.linalg.norm([dG_std[leg1], dG_std[leg2]])

            ddG_conv_df = convergence['complex'].copy()
            for tag in ['Forward', 'Backward']:
                ddG_conv_df[tag] = convergence[leg1][tag] - convergence[leg2][tag]
                ddG_conv_df[f'{tag}_Error'] = np.sqrt(
                    convergence[leg1][f'{tag}_Error'] ** 2 + convergence[leg2][f'{tag}_Error'] ** 2
                )
            ddG_conv_df.to_csv(pert_dir / f"{name}_convergence.csv", index=None)
            conv_ax = plot_convergence(ddG_conv_df)
            conv_ax.set_ylabel("$\Delta\Delta G$ (kcal/mol)")
            conv_ax.set_title(f"Convergence Analysis")
            conv_ax.figure.savefig(str(pert_dir / f"{name}_convergence.png"), dpi=300)

            if skip_gas:
                break

        with open(pert_dir / 'result.json', 'w') as f: 
            json.dump({"dG": dG, "dG_std": dG_std, 'ddG': ddG, 'ddG_std': ddG_std}, f, indent=4)
        
        self.logger.info(f"Finished - free energy evaulation. Results written to {pert_dir / 'result.json'}")

    def process_traj(self, protein_name: str, pert_name: str, remove_tmp: bool = True):
        """
        Processing trajectory of RBFE simulation endpoints: remove unphysical atoms, remove PBC, alignment 
        """
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        from rdkit import Chem
        import MDAnalysis as mda
        from MDAnalysis.analysis import align
        from MDAnalysis.transformations import wrap, unwrap, center_in_box
        from ...analysis.trajectory import compute_rmsd, plot_dihe_with_mol
        
        perturb_dir = self.rbfe_dir / protein_name / pert_name
        
        for leg in ['solvent', 'complex']:
            num_lambdas = len(glob.glob(os.path.join(perturb_dir, f'{leg}/lambda*/')))
            rmsd_fig, rmsd_ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)

            for lmd, resid in zip([0, num_lambdas - 1], [1, 2]):
                # Remove PBC and align
                in_top = os.path.join(perturb_dir, f'prep/{leg}.prmtop')
                in_trj = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/prod.mdcrd')

                out_pdb = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/prod_traj.pdb')
                out_trj = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/prod_traj.xtc')
                tmp_trj = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/prod_traj_tmp.xtc')
                trj_dir = os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/traj')
                if leg == 'complex' and not os.path.isdir(trj_dir):
                    os.mkdir(trj_dir)

                if leg == 'complex':
                    center_str = 'protein'
                    solute_str = f'protein or resid {resid}'
                    align_str = 'backbone'
                    ligand_str = f'resid {resid}'
                else:
                    center_str = f'resid {resid}'
                    solute_str = f'resid {resid}'
                    align_str = f'resid {resid}'
                    ligand_str = f'resid {resid}'

                u = mda.Universe(in_top, in_trj, format='NCDF')
                transformations = [
                    center_in_box(u.select_atoms(center_str)),
                    wrap(u.atoms),
                    unwrap(u.atoms)
                ]
                u.trajectory.add_transformations(*transformations)

                selection = u.select_atoms(solute_str)
                with mda.Writer(tmp_trj, n_atoms=selection.n_atoms) as W:
                    for i, ts in tqdm(enumerate(u.trajectory), total=len(u.trajectory), desc='Processing PBC'):
                        W.write(selection)
                        if i == 0:
                            selection.write(out_pdb)
                
                u_tmp = mda.Universe(out_pdb, tmp_trj)
                u_ref = mda.Universe(out_pdb, tmp_trj)
                alignment = align.AlignTraj(u_tmp, u_ref, select=align_str, in_memory=True)
                alignment.run()

                with mda.Writer(out_trj, n_atoms=u_tmp.atoms.n_atoms) as W:
                    for i, ts in tqdm(enumerate(u_tmp.trajectory), total=len(u_tmp.trajectory), desc='Align'):
                        W.write(u_tmp.atoms)
                        if leg == 'complex':
                            u_tmp.atoms.write(os.path.join(trj_dir, f'traj{i}.pdb'))

                u_out = mda.Universe(out_pdb, out_trj)
                if remove_tmp:
                    os.remove(tmp_trj)

                # rmsd
                rmsd_data = compute_rmsd(u_out, ligand_str, os.path.join(perturb_dir, f'{leg}/lambda{lmd}/prod/rmsd.txt'))
                rmsd_ax.plot(rmsd_data[:, 0], rmsd_data[:, 1], label=f'Lambda {lmd}')
                # dihedral
                char = 'A' if resid == 1 else 'B'
                ligand = Chem.SDMolSupplier(os.path.join(perturb_dir, f'prep/ligand{char}_renum.sdf'), removeHs=False)[0]
                plot_dihe_with_mol(u_out, ligand, os.path.join(perturb_dir, leg, f'lambda{lmd}', 'prod'))
            
            rmsd_ax.legend()
            rmsd_ax.set_ylabel(r'RMSD ($\mathrm{\mathring{A}}$)')
            rmsd_ax.set_xlabel('Time (ns)')
            rmsd_fig.savefig(os.path.join(perturb_dir, f'{leg}/rmsd.png'), dpi=300)
            plt.close(rmsd_fig)


# compatitable with old name
AmberRbfeProject = AmberLigandRbfeProject