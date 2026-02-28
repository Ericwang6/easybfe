from __future__ import annotations
import os
from pathlib import Path
import logging
import json
from typing import Union, Optional, TYPE_CHECKING

import numpy as np
import openmm.app as app
import openmm.unit as unit
import parmed

from .prep_utils import *
from .workflow import Step, Workflow
from ..core import Ligand, Protein
from ..parallel import run_func_parallel


if TYPE_CHECKING:
    from ..config import AmberSimulationConfig, AmberPlainMDConfig


logger = logging.getLogger(__name__)


def setup_plain_md(
    ligand: Union[Ligand | os.PathLike | None],
    protein: Union[Protein | os.PathLike | None],
    config: AmberSimulationConfig,
    wdir: os.PathLike
):  
    if ligand is None and protein is None:
        raise RuntimeError("Both ligand and protein are None.")

    # setup workding dir
    wdir = Path(wdir).expanduser().resolve()
    wdir.mkdir(exist_ok=True)
    basename = config.basename

    ffs = config.forcefields.copy()

    # setup systems
    modeller = app.Modeller(app.Topology(), [])

    ligand = ligand if isinstance(ligand, Ligand) or (ligand is None) else Ligand.from_directory(ligand)
    if ligand:
        ligand_pdb = ligand.to_openmm()
        modeller.add(ligand_pdb.topology, ligand_pdb.positions)
        ligand_dir = wdir / 'ligand'
        ligand.dump(ligand_dir)
        ffs.append(str(ligand_dir / f'{ligand.name}.xml'))
    
    protein = protein if isinstance(protein, Protein) or (protein is None) else Protein.from_pdb(protein)
    if protein:
        protein_pdb = protein.to_openmm()
        modeller.add(protein_pdb.topology, protein_pdb.positions)
        Path(wdir / f'protein.pdb').write_text(protein.pdb_string)
    
    ff = app.ForceField(*ffs)
    
    buffer = config.buffer / 10 * unit.nanometers
    box_vectors = computeBoxVectorsWithPadding(modeller.positions, buffer)
    modeller.positions = shiftToBoxCenter(modeller.positions, box_vectors)
    modeller.topology.setPeriodicBoxVectors(box_vectors)
    if not config.gas_phase:
        modeller.addSolvent(
            forcefield=ff,
            model=config.water_model,
            neutralize=True,
            ionicStrength=config.ionic_strength * unit.molar,
        )

    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False)
    parmed_struct = parmed.openmm.load_topology(modeller.topology, system, xyz=modeller.positions)
    
    # Handle Amber special SETTLE water 
    sanitize_water(parmed_struct)

    # HMR
    if config.do_hmr:
        hydrogen_mass_repartition(parmed_struct, config.hydrogen_mass, config.do_hmr_water)
    
    # output
    parmed_struct.save(str(wdir / f'{basename}.inpcrd'), overwrite=True)
    parmed_struct.save(str(wdir / f'{basename}.prmtop'), overwrite=True)
    parmed_struct.save(str(wdir / f'{basename}.pdb'), overwrite=True)

    # setup workflow
    steps = []
    for i, step_config in enumerate(config.workflow):
        # Step class expects AmberStepConfig object and creates mdin internally
        step = Step(config=step_config, wdir=wdir)
        steps.append(step)
    wf = Workflow(wdir, steps=steps, inpcrd=wdir / f'{basename}.inpcrd', prmtop=wdir / f'{basename}.prmtop')
    wf.create()
    return wf


def setup_plain_md_from_config(cfg: AmberPlainMDConfig):
    setup_plain_md(cfg.ligand, cfg.protein, cfg.simulation, cfg.output_dir)
    with open(cfg.output_dir / 'config.json', 'w') as f:
        json.dump(cfg.model_dump(mode="json"), f, indent=4)
