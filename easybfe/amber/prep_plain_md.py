import os
from pathlib import Path
import logging

import numpy as np
import openmm.app as app
import openmm.unit as unit
import parmed

from ..core.ligand import Ligand
from ..core.protein import Protein

from .prep_utils import *
from ..config import AmberSimulationConfig, AmberMdin
from .workflow import Step, Workflow


logger = logging.getLogger(__name__)


def setup_plain_md(
    ligand: Ligand,
    protein: Protein,
    config: AmberSimulationConfig,
    wdir: os.PathLike,
    basename: str = 'system'
):  
    if ligand is None and protein is None:
        raise RuntimeError("Both ligand and protein are None.")

    # setup workding dir
    wdir = Path(wdir).expanduser().resolve()
    wdir.mkdir(exist_ok=True)
    basename = wdir.stem if not basename else basename

    # dump ligand
    ligand.dump(wdir)

    # force field initialization
    ff = app.ForceField(*config.forcefields, str(wdir / f'{ligand.name}.xml'))

    # setup systems
    modeller = app.Modeller(app.Topology(), [])
    
    if protein:
        protein_pdb = protein.to_openmm()
        modeller.add(protein_pdb.topology, protein_pdb.positions)
    
    if ligand:
        ligand_pdb = app.PDBFile(str(wdir / f'{ligand.name}.pdb'))
        modeller.add(ligand_pdb.topology, ligand_pdb.positions)
    
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
        mdin = AmberMdin(cntrl=step_config.cntrl, wt=step_config.wt, rst=step_config.rst)
        step = Step(name=step_config.name, mdin=mdin, exec=step_config.exec)
        steps.append(step)
    wf = Workflow(wdir, steps=steps, inpcrd=wdir / f'{basename}.inpcrd', prmtop=wdir / f'{basename}.prmtop')
    wf.create()
    return True