import os
from typing import List, Dict, Any, Union, Tuple, Iterable, Optional
from pathlib import Path
import logging
import warnings

import numpy as np
import openmm as mm
import openmm.unit as unit 
import openmm.app as app 
import parmed
from rdkit import Chem

from .prep import (
    computeBoxVectorsWithPadding, shiftToBoxCenter, 
    hydrogen_mass_repartition, sanitize_water,
    PROTEIN_FF_XMLS, WATER_FF_XMLS
)
from .workflow import Step, create_groupfile_from_steps
from .settings import create_default_setting, AmberMdin, AmberWtSettings
from ..smff.utils import convert_to_xml
from ..cmd import init_logger


def merge_topology(
    topA: app.Topology, 
    topB: app.Topology, 
    posA: unit.Quantity,
    posB: unit.Quantity,
    mutated_residue_indices: Iterable[int]
):
    """
    Merge topologies to create a dual-topology used in Amber FEP simulation by
    merging given residues from B to A.

    Parameters
    ----------
    topA: openmm.app.Topology
        The topology of state A
    topB: openmm.app.Topology
        The topology of state B
    posA: openmm.unit.Quantity
        Atom positions of state A
    posB: openmm.unit.Quantity
        Atom positions of state B
    mutated_residue_indices: Iterable[int]
        Indices (starting from 0) of residues to be copied
    """
    new_top = app.Topology()
    new_pos = []

    new_top.setPeriodicBoxVectors(topA.getPeriodicBoxVectors())
    mut_residues = []

    for i in range(topA.getNumChains()):
        chainA = topA._chains[i]
        new_chain = new_top.addChain(chainA.id)
        for r in range(len(chainA)):
            resA = chainA._residues[r]
            new_res = new_top.addResidue(resA.name, new_chain, resA.id, resA.insertionCode)
            for atomA in resA.atoms():
                new_top.addAtom(atomA.name, atomA.element, new_res)
                new_pos.append(posA[atomA.index])
            if resA.index in mutated_residue_indices:
                chainB = topB._chains[i]
                resB = chainB._residues[r]
                mut = new_top.addResidue(resB.name, new_chain, resB.id, 'M')
                for atomB in resB.atoms():
                    new_top.addAtom(atomB.name, atomB.element, mut)
                    new_pos.append(posB[atomB.index])
                mut_residues.append((new_res, mut))
    
    # NOTE (Eric): This is a problem here. If topology A or B has non-standard residues, the merged topology
    # will miss some bonds. This shall be resolved by calling `Topology.loadBondDefinitions` prior to this function
    new_top.createStandardBonds()
    
    # OpenMM will falsely add a bond between the two mutated residues
    to_del = []
    for i, bo in enumerate(new_top.bonds()):
        ra, rb = bo[0].residue, bo[1].residue
        for muts in mut_residues:
            if (ra is muts[0] and rb is muts[1]) or (rb is muts[0] and ra is muts[1]):
                to_del.append(i)
                break
    for i in to_del[::-1]:
        new_top._bonds.pop(i)

    # But OpenMM will miss a bond between unmutated residue and its next residue
    # and a bond between mutated residue and its previous residue
    for resA, resB in mut_residues:
        atom1 = [at for at in resA.atoms() if at.name == 'C'][0]
        idx = resA.chain._residues.index(resA)
        atom2 = [at for at in resA.chain._residues[idx+2].atoms() if at.name == 'N'][0]
        new_top.addBond(atom1, atom2)

        atom1 = [at for at in resB.atoms() if at.name == 'N'][0]
        idx = resB.chain._residues.index(resB)
        atom2 = [at for at in resB.chain._residues[idx-2].atoms() if at.name == 'C'][0]
        new_top.addBond(atom1, atom2)
    
    muts = [(r1.index, r2.index) for r1, r2 in mut_residues]
    return new_top, new_pos, muts


def mk_merged_system(
    ff: app.ForceField,
    merged_top: app.Topology,
    merged_pos: Iterable,
    mutations: List[Tuple[int, int]],
):
    
    system = ff.createSystem(
        merged_top, 
        nonbondedMethod=app.PME,
        ignoreExternalBonds=True, 
        constraints=None, rigidWater=False
    )

    atoms = list(merged_top.atoms())
    
    def _is_bad_term(term, n):
        resIndices = [atoms[term[i]].residue.index for i in range(n)]
        flag = False
        for r1, r2 in mutations:
            if any(r == r1 for r in resIndices) and any(r == r2 for r in resIndices):
                flag = True
                break
        return flag

    # delete terms that falsely involve two mutated residues
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            for n in range(force.getNumBonds()):
                term = force.getBondParameters(n)
                if _is_bad_term(term, 2):
                    term[-1] = 0.0
                    force.setBondParameters(n, *term)
        elif isinstance(force, mm.HarmonicAngleForce):
            for n in range(force.getNumAngles()):
                term = force.getAngleParameters(n)
                if _is_bad_term(term, 3):
                    term[-1] = 0.0
                    force.setAngleParameters(n, *term)
        elif isinstance(force, mm.PeriodicTorsionForce):
            for n in range(force.getNumTorsions()):
                term = force.getTorsionParameters(n)
                if _is_bad_term(term, 4):
                    term[-1] = 0.0
                    force.setTorsionParameters(n, *term)
        elif isinstance(force, mm.NonbondedForce):
            for n in range(force.getNumExceptions()):
                term = force.getExceptionParameters(n)
                if _is_bad_term(term, 2):
                    term[2] = 0.0
                    term[-1] = 0.0
                    force.setExceptionParameters(n, *term)
        elif isinstance(force, mm.CMMotionRemover):
            continue
        else:
            raise NotImplementedError(f'Does not support: {force.__class__.__name__}')
    
    if isinstance(merged_pos[0], unit.Quantity):
        xyz = np.array([v.value_in_unit(unit.angstrom) for v in merged_pos])
    else:
        xyz = np.array(merged_pos)
    struct = parmed.openmm.load_topology(merged_top, system, xyz=xyz)

    # Delete the bad terms from parmed.Structure
    # we don't need to explicitly process nonbonded adjustments (1-4 scaling)
    # because they have been treated internally in parmed.openmm.load_topology,
    # i.e. terms with epsilon/chargeprod equals to zero are deleted
    for bond in struct.bonds:
        if bond.type.k == 0.0:
            bond.used = False
    struct.bonds.prune_unused()

    for angle in struct.angles:
        if angle.type.k == 0.0:
            angle.used = False
    struct.angles.prune_unused()

    for dihe in struct.dihedrals:
        if dihe.type.phi_k == 0.0:
            dihe.used = False
    struct.dihedrals.prune_unused()

    return struct


def find_mutations(topA: app.Topology, topB: app.Topology):
    """
    Find the indices (0-indexed) of mutated residues
    """
    mutations = []
    for chainA, chainB in zip(topA.chains(), topB.chains(), strict=True):
        for residueA, residueB in zip(chainA.residues(), chainB.residues(), strict=True):
            if residueA.name != residueB.name:
                mutations.append(residueA.index)
    return mutations


def setup_protein_fep_workflow(
    proteinA_pdb: os.PathLike,
    proteinB_pdb: os.PathLike,
    ligand_mol: Union[os.PathLike, Chem.Mol] = "",
    ligand_top: Union[os.PathLike, parmed.Structure] = "",
    wdir: os.PathLike = ".",
    protein_ff: str = 'ff14SB',
    water_ff: str = 'tip3p',
    config: Dict[str, Any] = dict(),
    use_charge_change: bool = True, 
    basename: str = 'protein',
    logger: Optional[logging.Logger] = None,
    overwrite: bool = True
):
    
    logger = init_logger() if logger is None else logger

    # Create working directory
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)

    # Determine Force fields
    if protein_ff.endswith('.xml'):
        protein_ff_xml = protein_ff
    else:
        protein_ff_xml = PROTEIN_FF_XMLS[protein_ff]
    
    if water_ff.endswith('.xml'):
        water_ff_xml = water_ff
    else:
        water_ff_xml = WATER_FF_XMLS[water_ff]
    xmls = [protein_ff_xml, water_ff_xml]
    
    # Read PDB
    pdbA = app.PDBFile(str(proteinA_pdb))
    pdbB = app.PDBFile(str(proteinB_pdb))
    topA = pdbA.topology
    topB = pdbB.topology
    posA = pdbA.positions
    posB = pdbB.positions

    # Find mutations
    mutated_residue_indices = find_mutations(topA, topB)
    
    # Add solvents & ions
    buffer = config.get("buffer", 15.0) /  10 * unit.nanometers
    boxVectors = computeBoxVectorsWithPadding(pdbA.positions, buffer)
    posA = shiftToBoxCenter(posA, boxVectors)
    posB = shiftToBoxCenter(posB, boxVectors)

    modeller = app.Modeller(topA, posA)

    if ligand_top and ligand_mol:
        if not isinstance(ligand_mol, Chem.Mol):
            ligand_mol = Chem.SDMolSupplier(ligand_mol, removeHs=False)[0]
        ligand_struct = parmed.load_file(str(ligand_top))
        ligand_struct.residues[0].name = 'MOL'
        ligand_struct.residues[0].resname = 'MOL'
        ligand_struct.coordinates = ligand_mol.GetConformer().GetPositions()
        convert_to_xml(ligand_struct, str(wdir / 'ligand.xml'), str(wdir / 'ligand_top.xml'))
        app.Topology.loadBondDefinitions(str(wdir / 'ligand_top.xml'))
        xmls.append(str(wdir / 'ligand.xml'))
        modeller.add(ligand_struct.topology, ligand_struct.positions)

    ff = app.ForceField(*xmls)
    modeller.topology.setPeriodicBoxVectors(boxVectors)
    modeller.addSolvent(
        forcefield=ff,
        model=water_ff,
        neutralize=True,
        ionicStrength=config.get('ionic_strength', 0.15) * unit.molar,
    )
    merged_top, merged_pos, mutations = merge_topology(
        modeller.topology, topB, modeller.positions, posB, mutated_residue_indices
    )
    
    ff = app.ForceField(*xmls)
    struct = mk_merged_system(ff, merged_top, merged_pos, mutations)
    sanitize_water(struct)
    if config.get("do_hmr", False):
        hydrogen_mass_repartition(struct, config.get('hydrogen_mass', 3.024))

    struct.save(str(wdir / f"{basename}.prmtop"), overwrite=overwrite)
    struct.save(str(wdir / f"{basename}.inpcrd"), overwrite=overwrite)
    app.PDBFile.writeFile(merged_top, struct.positions, str(wdir / f'{basename}.pdb'), keepIds=True)

    # Return soft-core atom indices, starting from 0
    scA, scB = [], []
    residues = list(merged_top.residues())
    for r1, r2 in mutations:
        for at in residues[r1].atoms():
            scA.append(at.index)
        for at in residues[r2].atoms():
            scB.append(at.index)

    mask = {
        "timask1": "@" + ",".join(str(x+1) for x in scA),
        "timask2": "@" + ",".join(str(x+1) for x in scB),
        "scmask1": "@" + ",".join(str(x+1) for x in scA),
        "scmask2": "@" + ",".join(str(x+1) for x in scB),
    }
    
    lambdas = config['lambdas']
    step_settings = config['workflow']
    step_names = []
    steps_total = []
    for i, setting in enumerate(step_settings):
        name = setting["name"]
        step_names.append(name)
        predef = setting["type"]

        if i == 0:
            assert predef == 'em', f"The first step must be energy minimization, not {predef}"
            prmtop = wdir / f"{basename}.prmtop"
            inpcrd = wdir / f"{basename}.inpcrd"
            pmemd_exec = 'pmemd.cuda'
            restart = False
        else:
            prmtop = None
            inpcrd = None
            assert predef != 'em', f"Only the first step can be energy minimization"
            pmemd_exec = 'pmemd.cuda.MPI'
            restart = True

        if i == len(step_settings) - 1:
            assert predef == 'prod', f"The last step must be production, not {predef}"
            setting['cntrl']['ifmbar'] = 1
        else:
            # Only the last step use mbar and gremd
            setting['cntrl']['ifmbar'] = 0

        # em and heat is not using velocities from the previous step
        restart = i > 0 and step_settings[i - 1]['type'] != 'em'

        # TI mask and lambdas
        setting['cntrl'].update(mask)
        setting['cntrl']['lambdas'] = lambdas
        
        if predef == 'em':
            # energy minimization
            predef_settings = create_default_setting(em=True, nvt=True, restraint=False, restart=restart, free_energy=True)
        elif predef == 'heat':
            # nvt - heating up (equilibrition)
            predef_settings = create_default_setting(em=False, nvt=True, restraint=True, restart=restart, free_energy=True)
            predef_settings['cntrl']['nmropt'] = 1
        elif predef == "pres":
            # npt - pressurize (equilibrition)
            predef_settings = create_default_setting(em=False, nvt=False, restraint=True, restart=restart, free_energy=True)
        elif predef == "nvt":
            # production with NVT, not frequently used
            predef_settings = create_default_setting(em=False, nvt=True, restraint=False, restart=restart, free_energy=True)
        else:
            # production with NPT
            predef_settings = create_default_setting(em=False, nvt=False, restraint=False, restart=restart, free_energy=True)
            if predef != 'prod':
                warnings.warn("Unrecognized pre-defined setting types. NPT production pre-defined settings are used with user-specified &cntrl options")

        predef_settings['cntrl'].update(setting['cntrl'])
        
        # create step for each lambda
        steps = []
        for n, clambda in enumerate(lambdas):
            lambda_dir = wdir / f'lambda{n}'
            lambda_dir.mkdir(exist_ok=True)
            step_dir = lambda_dir / name
            step_dir.mkdir(exist_ok=True)

            predef_settings['cntrl']['clambda'] = clambda
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
                name=name, wdir=step_dir, mdin=mdin, exec=pmemd_exec,
                prmtop=prmtop, inpcrd=inpcrd,
            )

            # Link steps
            # Energy minimization follow the precedure that lambda i starts from lambda i-1
            if n > 0 and i == 0:
                step.link_prev_step(steps[-1])
            # The rest starts from its previous step
            if i > 0:
                step.link_prev_step(steps_total[-1][n])
            
            # Generate command to run this step
            step.create()                

            steps.append(step)

        steps_total.append(steps)
    
    # Use groupfile and MPI to run steps except energy minimization
    for i in range(1, len(step_settings)):
        create_groupfile_from_steps(
            steps_total[i], 
            dirname=wdir,
            fpath=os.path.join(wdir, f'{step_names[i]}.groupfile')
        )

    # Write run.sh
    with open(Path(__file__).parent / 'run.sh.template') as f:
        script = f.read()
    
    script = script.replace('@NUM_LAMBDA', str(len(lambdas)))
    script = script.replace('@STAGES', '({})'.format(' '.join(f'"{stage}"' for stage in step_names)))
    script = script.replace('@MD_EXEC', 'pmemd.cuda.MPI')

    with open(wdir / 'run.sh', 'w') as f:
        f.write(script)