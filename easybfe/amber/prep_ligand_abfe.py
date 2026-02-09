import json
import os
from pathlib import Path
import logging

import math
import numpy as np
import openmm.app as app
import openmm.unit as unit
import parmed
from dataclasses import dataclass

from .prep_utils import *
from ..config import AmberFepSimulationConfig, AmberMdin, AmberRstSettings
from .workflow import Step, create_groupfile_from_steps
from ..smff.utils import OpenmmXML
from ..core import Ligand, Protein


logger = logging.getLogger(__name__)


def compute_bond(pos0, pos1):
    dx, dy, dz = pos1[0]-pos0[0], pos1[1]-pos0[1], pos1[2]-pos0[2]
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def compute_angle(pos0, pos1, pos2):
    """
    Compute the angle between three positions.
    
    Computes the angle at pos1 formed by the vectors pos0->pos1 and pos1->pos2.
    
    Parameters
    ----------
    pos0 : array-like
        First position (x, y, z).
    pos1 : array-like
        Central position (x, y, z).
    pos2 : array-like
        Third position (x, y, z).
    
    Returns
    -------
    float
        Angle in degrees, in the range (0, 180).
    """
    # Convert to numpy arrays for vector operations
    pos0 = np.array(pos0)
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    
    # Compute vectors
    vec1 = pos0 - pos1  # Vector from pos1 to pos0
    vec2 = pos2 - pos1  # Vector from pos1 to pos2
    
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute angle: zero-length vector")
    
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2
    
    # Compute angle using dot product
    cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


def compute_dihedral(pos0, pos1, pos2, pos3):
    """
    Compute the dihedral angle (torsion angle) between four positions.
    
    Computes the dihedral angle defined by the four positions pos0-pos1-pos2-pos3.
    This is the angle between the planes defined by pos0-pos1-pos2 and pos1-pos2-pos3.
    
    Parameters
    ----------
    pos0 : array-like
        First position (x, y, z).
    pos1 : array-like
        Second position (x, y, z).
    pos2 : array-like
        Third position (x, y, z).
    pos3 : array-like
        Fourth position (x, y, z).
    
    Returns
    -------
    float
        Dihedral angle in degrees, in the range (-180, 180).
    """
    # Convert to numpy arrays for vector operations
    pos0 = np.array(pos0)
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    pos3 = np.array(pos3)
    
    # Compute vectors
    vec1 = pos1 - pos0  # Vector from pos0 to pos1
    vec2 = pos2 - pos1  # Vector from pos1 to pos2
    vec3 = pos3 - pos2  # Vector from pos2 to pos3
    
    # Compute cross products to get normal vectors to the planes
    cross1 = np.cross(vec1, vec2)  # Normal to plane pos0-pos1-pos2
    cross2 = np.cross(vec2, vec3)  # Normal to plane pos1-pos2-pos3
    
    # Normalize cross products
    norm1 = np.linalg.norm(cross1)
    norm2 = np.linalg.norm(cross2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute dihedral: degenerate geometry (atoms are collinear)")
    
    cross1 = cross1 / norm1
    cross2 = cross2 / norm2
    
    # Normalize vec2 for sign determination
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec2 == 0:
        raise ValueError("Cannot compute dihedral: zero-length central bond")
    vec2_norm = vec2 / norm_vec2
    
    # Compute sin and cos of dihedral angle
    cos_angle = np.clip(np.dot(cross1, cross2), -1.0, 1.0)
    sin_angle = np.dot(np.cross(cross1, cross2), vec2_norm)
    
    # Use atan2 for proper sign determination (returns angle in [-pi, pi])
    dihedral_rad = math.atan2(sin_angle, cos_angle)
    dihedral_deg = math.degrees(dihedral_rad)
    
    return dihedral_deg


@dataclass
class BoreschRestraint:
    protein_anchors: tuple
    ligand_anchors: tuple
    rst_wts: tuple
    rst_vals: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def compute_rst_vals(self, protein_positions: np.ndarray, ligand_positions: np.ndarray):
        """
        Compute Boresch restraint values from protein and ligand positions.
        
        Computes the six geometric parameters (r0, alpha0, theta0, gamma0, beta0, phi0)
        that define the Boresch restraints based on the current positions of the anchor atoms.
        
        Parameters
        ----------
        protein_positions : np.ndarray
            Array of protein atom positions, shape (n_protein_atoms, 3).
        ligand_positions : np.ndarray
            Array of ligand atom positions, shape (n_ligand_atoms, 3).
        
        Returns
        -------
        tuple[float, float, float, float, float, float]
            Tuple containing (r0, alpha0, theta0, gamma0, beta0, phi0) in degrees for angles/dihedrals.
            r0 is in Angstroms.
        """
        l1, l2, l3 = self.ligand_anchors
        p1, p2, p3 = self.protein_anchors
        
        # Extract positions for anchor atoms
        pos_l1 = ligand_positions[l1]
        pos_l2 = ligand_positions[l2]
        pos_l3 = ligand_positions[l3]
        pos_p1 = protein_positions[p1]
        pos_p2 = protein_positions[p2]
        pos_p3 = protein_positions[p3]
        
        # Compute distance: L1-P1
        r0 = compute_bond(pos_l1, pos_p1)
        
        # Compute angle: P1-L1-L2
        alpha0 = compute_angle(pos_p1, pos_l1, pos_l2)
        
        # Compute angle: P2-P1-L1
        theta0 = compute_angle(pos_p2, pos_p1, pos_l1)
        
        # Compute dihedral: P1-L1-L2-L3
        gamma0 = compute_dihedral(pos_p1, pos_l1, pos_l2, pos_l3)
        
        # Compute dihedral: P2-P1-L1-L2
        beta0 = compute_dihedral(pos_p2, pos_p1, pos_l1, pos_l2)
        
        # Compute dihedral: P3-P2-P1-L1
        phi0 = compute_dihedral(pos_p3, pos_p2, pos_p1, pos_l1)
        
        vals = (r0, alpha0, theta0, gamma0, beta0, phi0)
        self.rst_vals = vals
        return vals

    def make_rst(self, offset: int):
        """Construct multiple Boresch restraints.
        
        Parameters
        ----------
        offset : int
            Offset to add to protein atom indices (typically number of ligand atoms).
        
        Returns
        -------
        list[AmberRstSettings]
            List of restraint settings for distance, angles, and dihedrals.
        """
        l1, l2, l3 = self.ligand_anchors
        p1, p2, p3 = self.protein_anchors
        r0, alpha0, theta0, gamma0, beta0, phi0 = self.rst_vals
        rk, alphak, thetak, gammak, betak, phik = self.rst_wts
        
        # Adjusted boundaries for dihedrals (accounts for periodicity)
        dih11, dih14 = gamma0 - 180.0, gamma0 + 180.0
        dih21, dih24 = beta0 - 180.0, beta0 + 180.0
        dih31, dih34 = phi0 - 180.0, phi0 + 180.0
        
        # Build the list of restraint settings
        # Note: iat uses 1-based indexing, so add 1 to all atom indices
        rst_list = [
            # Distance restraint: L1-P1
            AmberRstSettings(iat=[l1+1, p1+offset+1], r1=0.0, r2=r0, r3=r0, r4=99.0, rk2=rk, rk3=rk),
            # Angle restraint: P1-L1-L2
            AmberRstSettings(iat=[p1+offset+1, l1+1, l2+1], r1=0.0, r2=alpha0, r3=alpha0, r4=180.0, rk2=alphak, rk3=alphak),
            # Angle restraint: P2-P1-L1
            AmberRstSettings(iat=[p2+offset+1, p1+offset+1, l1+1], r1=0.0, r2=theta0, r3=theta0, r4=180.0, rk2=thetak, rk3=thetak),
            # Dihedral restraint: P1-L1-L2-L3
            AmberRstSettings(iat=[p1+offset+1, l1+1, l2+1, l3+1], r1=dih11, r2=gamma0, r3=gamma0, r4=dih14, rk2=gammak, rk3=gammak),
            # Dihedral restraint: P2-P1-L1-L2
            AmberRstSettings(iat=[p2+offset+1, p1+offset+1, l1+1, l2+1], r1=dih21, r2=beta0, r3=beta0, r4=dih24, rk2=betak, rk3=betak),
            # Dihedral restraint: P3-P2-P1-L1
            AmberRstSettings(iat=[p3+offset+1, p2+offset+1, p1+offset+1, l1+1], r1=dih31, r2=phi0, r3=phi0, r4=dih34, rk2=phik, rk3=phik)
        ]
        
        return rst_list


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
    box_vectors = computeBoxVectorsWithPadding(modeller.positions, buffer)
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
    mode = 'abfe' if restraints is None else 'abfe_restr'
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
    steps = []
    steps_total = []
    for i, step_config in enumerate(config.workflow):
        prmtop = wdir / f'{basename}.prmtop' if i == 0 else None
        inpcrd = wdir / f'{basename}.prmtop' if i == 0 else None
        step_config.rst += rst_settings
        step_config.cntrl = step_config.cntrl.model_copy(update=mask)

        for n, clambda in enumerate(config.lambdas):
            lambda_dir = wdir / f'lambda{n}'
            lambda_dir.mkdir(exist_ok=True)
            step_dir = lambda_dir / step_config.name
            step_dir.mkdir(exist_ok=True)

            step_config.cntrl.clambda = clambda
            mdin = AmberMdin(cntrl=step_config.cntrl, wt=step_config.wt, rst=step_config.rst)
            step = Step(name=step_config.name, wdir=step_dir, mdin=mdin, exec=step_config.exec, prmtop=prmtop, inpcrd=inpcrd)
        
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
    for i in range(1, len(config.workflow)):
        create_groupfile_from_steps(
            steps_total[i], 
            dirname=wdir,
            fpath=os.path.join(wdir, f'{config.workflow[i].name}.groupfile')
        )

    # Write run.sh
    with open(Path(__file__).parent / 'run.sh.template') as f:
        script = f.read()
    
    script = script.replace('@NUM_LAMBDA', str(len(config.lambdas)))
    script = script.replace('@STAGES', '({})'.format(' '.join(f'"{step_config.name}"' for step_config in config.workflow)))
    script = script.replace('@MD_EXEC', 'pmemd.cuda.MPI')
    script = script.replace('@EM_NAME', steps_total[0][0].name)

    with open(wdir / 'run.sh', 'w') as f:
        f.write(script)

    return True


def setup_ligand_abfe(
    ligand: Ligand, protein: Protein,
    leg_configs: dict[str, AmberFepSimulationConfig],
    restraints: BoreschRestraint,
    output_dir: os.PathLike,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    setup_ligand_abfe_leg(ligand, None, leg_configs['solvent'], output_dir/'solvent', basename='system')
    setup_ligand_abfe_leg(ligand, protein, leg_configs['complex'], output_dir/'complex', restraints=restraints, basename='system')
    setup_ligand_abfe_leg(
        ligand, protein, 
        leg_configs['restraint'], output_dir/'restraint',
        duplicate_ligand=True, restraints=restraints, basename='system'
    )
