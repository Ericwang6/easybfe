import os
from pathlib import Path
from typing import Dict, Any, List, Union
import warnings
import parmed
from .workflow import Step, Workflow
from .settings import AmberMdin, create_default_setting, AmberWtSettings
from ..smff.utils import convert_to_xml
from .prep import computeBoxVectorsWithPadding, shiftToBoxCenter, hydrogen_mass_repartition
import openmm.app as app
import openmm.unit as unit


PROTEIN_FF_XMLS = {
    'ff14SB': 'amber14/protein.ff14SB.xml'
}

WATER_FF_XMLS = {
    'tip3p': 'amber14/tip3p.xml'
}

def create_system(
    protein_pdb: os.PathLike = "",
    ligand_prmtop: os.PathLike = "",
    ligand_inpcrd: os.PathLike = "",
    output_dir: os.PathLike = ".",
    protein_ff: Union[str, List[str]] = 'ff14SB',
    water_ff: str = 'tip3p',
    box_shape: str = 'cube',
    buffer: float = 20.0,
    ionic_strength: float = 0.15,
    do_hmr: bool = True,
    do_hmr_water: bool = False,
    overwrite: bool = False
):
    has_protein = protein_pdb != ''
    has_ligand = ligand_inpcrd != '' and ligand_inpcrd != ''
    if (not has_protein) and (not has_ligand):
        raise RuntimeError("No protein or ligand is provided")
    
    ffs = []
    modeller = app.Modeller(app.Topology(), [])
    if has_protein:
        pdb = app.PDBFile(str(protein_pdb))
        modeller.add(pdb.topology, pdb.positions)
        if isinstance(protein_ff, list):
            for ff in protein_ff:
                ffs.append(PROTEIN_FF_XMLS.get(ff, ff))
        else:
            ffs.append(PROTEIN_FF_XMLS.get(protein_ff, protein_ff))
    if has_ligand:
        ligand_xml = os.path.join(output_dir, 'ligand.xml')
        ligand_struct = parmed.load_file(str(ligand_prmtop), xyz=str(ligand_inpcrd))
        convert_to_xml(ligand_struct, ligand_xml)
        modeller.add(ligand_struct.topology, ligand_struct.positions)
        ffs.append(ligand_xml)
    
    ffs.append(WATER_FF_XMLS[water_ff])
    ff = app.ForceField(*ffs)
    
    boxVectors = computeBoxVectorsWithPadding(
        modeller.positions, 
        buffer / 10.0 * unit.nanometers,
        box_shape
    )
    modeller.positions = shiftToBoxCenter(modeller.positions, boxVectors)
    modeller.topology.setPeriodicBoxVectors(boxVectors)
    modeller.addSolvent(
        forcefield=ff,
        model=water_ff,
        neutralize=True,
        ionicStrength=ionic_strength * unit.molar
    )

    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False)
    system_struct = parmed.openmm.load_topology(
        modeller.topology,
        system,
        xyz=modeller.positions
    )
    if do_hmr:
        hydrogen_mass_repartition(system_struct, dowater=do_hmr_water)

    system_struct.save(os.path.join(output_dir, 'system.pdb'), overwrite=overwrite)

    for residx, residue in enumerate(system_struct.residues):
        # Amber uses WAT to identify SETTLE for waters
        if residue.name == 'HOH':
            residue.name = 'WAT'
        if residue.name != 'HOH' and residue.name != 'WAT':
            continue
        atom_dict = {atom.name: atom for atom in residue.atoms}
        hhtype = parmed.BondType(k=553.000, req=1.514, list=system_struct.bond_types)
        system_struct.bond_types.append(hhtype)
        hh_bond = parmed.Bond(atom_dict['H1'], atom_dict['H2'], hhtype)
        system_struct.bonds.append(hh_bond)
    
    to_del = []
    for i, angle in enumerate(system_struct.angles):
        resname = angle.atom1.residue.name
        if resname == 'HOH' or resname == 'WAT':
            to_del.append(i)
    for item in reversed(to_del):
        system_struct.angles.pop(item)

    system_struct.save(os.path.join(output_dir, 'system.prmtop'), overwrite=overwrite)
    system_struct.save(os.path.join(output_dir, 'system.inpcrd'), overwrite=overwrite)


def create_workflow(
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    step_settings: List[Dict[str, Any]],
    wdir: os.PathLike,
    exec: str = 'pmemd.cuda'
):
    steps = []
    wdir = Path(wdir).resolve()
    for i, setting in enumerate(step_settings):
        name = setting["name"]
        predef = setting["type"]
        # em and heat is not using velocities from the previous step
        restart = i > 0 and step_settings[i - 1]['type'] != 'em'
        
        if predef == 'em':
            # energy minimization
            predef_settings = create_default_setting(em=True, nvt=True, restraint=False, restart=restart)
        elif predef == 'heat':
            # nvt - heating up (equilibrition)
            predef_settings = create_default_setting(em=False, nvt=True, restraint=True, restart=restart)
            predef_settings['cntrl']['nmropt'] = 1
        elif predef == "pres":
            # npt - pressurize (equilibrition)
            predef_settings = create_default_setting(em=False, nvt=False, restraint=True, restart=restart)
        elif predef == "nvt":
            # production with NVT
            predef_settings = create_default_setting(em=False, nvt=True, restraint=False, restart=restart)
        else:
            # production with NPT
            predef_settings = create_default_setting(em=False, nvt=False, restraint=False, restart=restart)
            if predef != 'prod':
                warnings.warn("Unrecognized pre-defined setting types. NPT production pre-defined settings are used with user-specified &cntrl options")

        predef_settings['cntrl'].update(setting['cntrl'])
        mdin = AmberMdin(**predef_settings)
        if predef == 'heat':
            # heating schedule: first half heat up, second half equilibriation
            mdin.wt.append(
                AmberWtSettings(
                    type='TEMP0', 
                    istep1=1, 
                    istep2=mdin.cntrl.nstlim // 2, 
                    value1=mdin.cntrl.tempi, 
                    value2=mdin.cntrl.temp0
                )
            )

        step = Step(
            name=name,
            wdir=wdir / name,
            mdin=mdin,
            exec=exec
        )
        steps.append(step)

    wf = Workflow(wdir, prmtop, inpcrd, steps)
    return wf

