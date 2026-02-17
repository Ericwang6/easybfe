import os
from pathlib import Path
import logging
from typing import Optional

import numpy as np
import openmm.app as app
import openmm.unit as unit
import parmed

from .prep_utils import *
from ..config import AmberFepSimulationConfig, AmberWtSettings
from ..config.amber.abfe import AmberAbfeConfig, BoreschRestraintGeneratorConfig
from .workflow import Step, Workflow, create_script_for_workflows
from ..core import Ligand, Protein
from .boresch import BoreschRestraint, compute_boresch_energy, get_boresch_finder
from ..parallel import run_func_parallel


logger = logging.getLogger(__name__)


def setup_ligand_abfe_leg(
    ligand: Ligand, 
    protein: Protein | None,
    config: AmberFepSimulationConfig,
    wdir: os.PathLike,
    duplicate_ligand: bool = False,
    restraints: BoreschRestraint | None = None,
    basename: str | None = None,
):  
    # setup workding dir
    wdir = Path(wdir).expanduser().resolve()
    wdir.mkdir(exist_ok=True)
    basename = wdir.stem if not basename else basename

    ligand.dump(wdir)
    ligand_pdb = app.PDBFile(str(wdir / f'{ligand.name}.pdb'))

    # charges
    ligandA_charge = compute_net_charge_from_openmm_system(
        app.ForceField(str(wdir / f'{ligand.name}.xml')).createSystem(ligand_pdb.topology)
    )

    # force field initialization
    ff = app.ForceField(*config.forcefields, str(wdir / f'{ligand.name}.xml'))

    # setup systems
    modeller = app.Modeller(app.Topology(), [])
    modeller.add(ligand_pdb.topology, ligand_pdb.positions)
    
    # use for resolve restraints in protein-ligand complex
    if duplicate_ligand:
        modeller.add(ligand_pdb.topology, ligand_pdb.positions)

    if protein:
        protein_openmm = protein.to_openmm()
        modeller.add(protein_openmm.topology, protein_openmm.positions)
    
    buffer = config.buffer / 10 * unit.nanometers
    box_vectors = computeBoxVectorsWithPadding(modeller.positions, buffer, config.box_shape)
    modeller.positions = shiftToBoxCenter(modeller.positions, box_vectors)
    modeller.topology.setPeriodicBoxVectors(box_vectors)
    assert not config.gas_phase, 'Gas-phase ABFE are ill-defined!'
    modeller.addSolvent(
        forcefield=ff,
        model=config.water_model,
        neutralize=True,
        ionicStrength=config.ionic_strength * unit.molar
    )

    # generate masks
    mode = 'abfe_restr' if duplicate_ligand else 'abfe'
    num_ligand_atoms = len(list(ligand_pdb.topology.atoms()))
    mask = generate_amber_mask(num_ligand_atoms, -1, {}, mode=mode)

    # alchemical water
    d_charge = -ligandA_charge
    alchem_waters, rst_settings = [], []
    if d_charge != 0:
        logger.info(f"Perturbation invoves charge change {int(ligandA_charge)} -> 0")
        if config.use_charge_change and (not config.gas_phase):
            scIndices = mask['scmask1'].strip("'")[1:].split(',') + mask['scmask2'].strip("'")[1:].split(',')
            scIndices = [int(x) - 1 for x in scIndices if x]
            alchem_water_info = do_co_alchemical_water(modeller, d_charge, scIndices)
            rst_settings = set_alchemical_water_restraints(modeller, scIndices, alchem_water_info)
            alchem_waters = [] if config.use_settle_for_alchemical_water else alchem_water_info['alchemical_water_residues']
            mask = generate_amber_mask(num_ligand_atoms, -1, {}, alchem_water_info, mode=mode)
        else:
            logger.warning("Charge change not enabled. Results are not trustworthy.")
    
    # apply boresch restraints
    if restraints:
        assert protein is not None, 'Boresch restraints must be used with protein'
        ligand_pos_angstrom = np.array([[p.x, p.y, p.z] for p in ligand_pdb.positions]) * 10
        protein_pos_angstrom = np.array([[p.x, p.y, p.z] for p in protein_openmm.positions]) * 10
        restraints.compute_rst_vals(protein_pos_angstrom, ligand_pos_angstrom)
        rst_settings += restraints.make_rst(
            offset=num_ligand_atoms if not duplicate_ligand else num_ligand_atoms*2
        )
        
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False)
    parmed_struct = parmed.openmm.load_topology(modeller.topology, system, xyz=modeller.positions)
    
    # Handle Amber special SETTLE water 
    sanitize_water(parmed_struct, 'ALW', alchem_waters)

    # HMR
    if config.do_hmr:
        hydrogen_mass_repartition(parmed_struct, config.hydrogen_mass, config.do_hmr_water)
    
    # output
    parmed_struct.save(str(wdir / f'{basename}.inpcrd'), overwrite=True)
    parmed_struct.save(str(wdir / f'{basename}.prmtop'), overwrite=True)
    parmed_struct.save(str(wdir / f'{basename}.pdb'), overwrite=True)

    # setup workflow
    workflows = []
    for n, clambda in enumerate(config.lambdas):
        steps = []
        for step_template in config.workflow:
            # Build a per-lambda AmberStepConfig with updated cntrl / wt / rst
            update = mask.copy()
            update.update({
                "clambda": clambda, 
                'ntwx': 10*step_template.cntrl.ntwx if (config.reduce_storage and n > 0 and n < (len(config.lambdas)-1)) else step_template.cntrl.ntwx
            })
            step_lambda_cntrl = step_template.cntrl.model_copy(update=update)
            rst = step_template.rst + rst_settings
            wt = step_template.wt + [AmberWtSettings(type="DUMPFREQ", istep1=step_lambda_cntrl.ofreq)]

            step_config = step_template.model_copy()
            step_config.cntrl = step_lambda_cntrl
            step_config.rst = rst
            step_config.wt = wt

            step = Step(config=step_config)
            steps.append(step)

        lambda_dir = wdir / f'lambda{n}'
        prmtop = wdir / f'{basename}.prmtop'
        inpcrd = wdir / f'{basename}.inpcrd'
        wf = Workflow(wdir=lambda_dir, prmtop=prmtop, inpcrd=inpcrd, steps=steps)
        wf.create()
        workflows.append(wf)
    
    # Use groupfile and MPI to run steps except energy minimization
    create_script_for_workflows(workflows, wdir, config.num_procs)

    return True


def setup_ligand_abfe(
    ligand: Ligand,
    protein: Protein,
    leg_configs: dict[str, AmberFepSimulationConfig],
    output_dir: os.PathLike,
    restraints: BoreschRestraint | BoreschRestraintGeneratorConfig | None = None,
):
    if restraints is None:
        restraints = BoreschRestraintGeneratorConfig()
    if not isinstance(restraints, BoreschRestraint):
        boresch_config = restraints
        finder_cls = get_boresch_finder(boresch_config.algorithm)
        finder_kwargs = {
            "protein": protein,
            "ligand": ligand,
            "wts": tuple(boresch_config.rst_wts),
            **boresch_config.options,
        }
        finder = finder_cls(**finder_kwargs)
        restraints = finder.find()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    setup_ligand_abfe_leg(ligand, None, leg_configs['solvent'], output_dir/'solvent', basename='system')
    setup_ligand_abfe_leg(ligand, protein, leg_configs['complex'], output_dir/'complex', restraints=restraints, basename='system')
    setup_ligand_abfe_leg(
        ligand, protein, 
        leg_configs['restraint'], output_dir/'restraint',
        duplicate_ligand=True, restraints=restraints, basename='system'
    )
    boresch_fe = compute_boresch_energy(restraints.rst_vals, restraints.rst_wts)
    (output_dir / 'boresch.dat').write_text(str(boresch_fe))


def _setup_ligand_abfe_one(
    ligand_path: Path,
    protein: Optional[Protein],
    leg_configs: dict[str, AmberFepSimulationConfig],
    boresch_config: BoreschRestraintGeneratorConfig,
    output_dir: Path,
) -> None:
    """Load ligand from path and call setup_ligand_abfe (used for batch runs)."""
    ligand = Ligand.from_directory(ligand_path)
    setup_ligand_abfe(
        ligand=ligand,
        protein=protein,
        leg_configs=leg_configs,
        restraints=boresch_config,
        output_dir=output_dir,
    )


def setup_ligand_abfe_from_config(
    config: AmberAbfeConfig,
    num_procs: Optional[int] = None,
) -> None:
    """Run setup_ligand_abfe from an :class:`AmberAbfeConfig`.

    If :attr:`AmberAbfeConfig.ligand_batch` is set, runs one setup per ligand in
    parallel; otherwise uses :attr:`AmberAbfeConfig.ligand` and writes directly
    to :attr:`AmberAbfeConfig.output_dir`.
    """
    assert config.protein is not None, "AmberAbfeConfig.protein must be set"
    assert config.output_dir is not None, "AmberAbfeConfig.output_dir must be set"

    leg_configs = {
        "complex": config.complex,
        "solvent": config.solvent,
        "restraint": config.restraint,
    }
    base_out = Path(config.output_dir)
    base_out.mkdir(exist_ok=True)
    protein = Protein.from_pdb(config.protein, name=config.protein.stem)

    if config.ligand_batch is not None and config.ligand is not None:
        raise ValueError(
            "AmberAbfeConfig must set either ligand or ligand_batch, not both"
        )

    if config.ligand_batch is not None:
        nprocs = num_procs if num_procs is not None else -1
        args_list = [
            (path, protein, leg_configs, config.boresch, base_out / path.stem)
            for path in config.ligand_batch
        ]
        run_func_parallel(
            _setup_ligand_abfe_one,
            args_list,
            nprocs=nprocs,
            unpack_args=True,
            desc="setup_ligand_abfe",
        )
    else:
        if config.ligand is None:
            raise ValueError("AmberAbfeConfig must set either ligand or ligand_batch")
        ligand = Ligand.from_directory(config.ligand)
        setup_ligand_abfe(
            ligand=ligand,
            protein=protein,
            leg_configs=leg_configs,
            restraints=config.boresch,
            output_dir=base_out,
        )