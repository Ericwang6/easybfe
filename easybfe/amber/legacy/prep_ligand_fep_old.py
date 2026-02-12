from ntpath import isfile
import os
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
import math
from io import StringIO

import numpy as np
import openmm.app as app
import openmm.unit as unit
import parmed
from rdkit import Chem

from .prep_utils import *
from ..config import AmberFepSimulationConfig, AmberMdin
from .workflow import Step, create_groupfile_from_steps
from ..smff.utils import OpenmmXML
from ..ligand import Ligand


logger = logging.getLogger(__name__)


def _check_renumber_mapping_dict(mapping_dict: dict[int, int], natoms: int):
    ref_list = list(range(natoms))
    old_numbers = list(mapping_dict.keys())
    old_numbers.sort()
    assert ref_list == old_numbers, f"Old numbers not valid"
    new_numbers = list(mapping_dict.values())
    new_numbers.sort()
    assert ref_list == new_numbers, f"New numbers not valid"


def renumber_rdkit_mol(old_mol: Chem.Mol, mapping_dict: dict[int, int]):
    _check_renumber_mapping_dict(mapping_dict, old_mol.GetNumAtoms())
    new_mol = Chem.RWMol()
    # new mapping -> old mapping
    rev_dict = {v:k for k,v in mapping_dict.items()}
    for k in range(old_mol.GetNumAtoms()):
        atom = old_mol.GetAtomWithIdx(rev_dict[k])
        new_mol.AddAtom(atom)
    
    for bond in old_mol.GetBonds():
        new_begin_idx = mapping_dict[bond.GetBeginAtomIdx()]
        new_end_idx = mapping_dict[bond.GetEndAtomIdx()]
        new_mol.AddBond(new_begin_idx, new_end_idx, bond.GetBondType())
    conf = old_mol.GetConformer()
    new_conf = Chem.Conformer(new_mol.GetNumAtoms())
    for old_idx, new_idx in mapping_dict.items():
        pos = conf.GetAtomPosition(old_idx)
        new_conf.SetAtomPosition(new_idx, pos)
    new_mol.AddConformer(new_conf)
    new_mol = new_mol.GetMol()
    new_mol.SetProp('_Name', old_mol.GetProp('_Name'))
    return new_mol


def mk_renumber_mapping_from_cc(cc_list: list[int], natoms: int):
    mapping = {}
    for i, cc in enumerate(cc_list):
        mapping[cc] = i
    for i in range(natoms):
        if i not in cc_list:
            mapping[i] = len(mapping)
    return mapping


def renumber_ligand_pdb(input_pdb: str | Path, mapping: dict[int, int], output_pdb: os.PathLike = None, rename_residue: str | None = None):
    """
    Renumber atoms in a ligand PDB file according to a mapping dictionary.
    
    This function reads a PDB file using OpenMM, renumbers the atoms according
    to the provided mapping (old index -> new index), and writes the renumbered
    structure to a new PDB file. The function assumes the PDB contains only one
    chain and one residue (a single ligand molecule).
    
    Parameters
    ----------
    pdb_file : os.PathLike
        Path to the input PDB file to renumber.
    mapping : dict[int, int]
        dictionary mapping old atom indices (keys) to new atom indices (values).
        Must be a valid permutation of all atom indices (0 to n-1).
    output_pdb : os.PathLike
        Path to the output PDB file where the renumbered structure will be saved.
    rename_residue : str, optional
        If provided, rename the ligand residue to this name in the output PDB.
        If None, the original residue name is preserved.
    
    Raises
    ------
    AssertionError
        If the mapping dictionary is invalid (does not contain all indices or
        is not a valid permutation), or if the PDB file does not contain exactly
        one chain and one residue.
    
    Notes
    -----
    The mapping dictionary should map old indices to new indices. For example,
    if atom 0 should become atom 2, then mapping[0] = 2.
    
    The function validates the mapping using :func:`_check_renumber_mapping_dict`
    before proceeding with the renumbering.
    
    This function is designed specifically for ligand PDB files that contain a
    single molecule (one chain, one residue). For more complex structures, use
    a different function.
    
    Examples
    --------
    >>> # Swap first two atoms and rename residue
    >>> mapping = {0: 1, 1: 0, 2: 2, 3: 3}
    >>> renumber_ligand_pdb('input.pdb', mapping, 'output.pdb', rename_residue='MOL')
    """
    # Read the PDB file
    if os.path.isfile(input_pdb):
        pdb = app.PDBFile(str(input_pdb))
    else:
        inp = StringIO()
        inp.write(input_pdb)
        pdb = app.PDBFile(inp)

    topology = pdb.topology
    positions = pdb.positions
    
    # Assert that there is only one chain and one residue
    chains = list(topology.chains())
    assert len(chains) == 1, f"Expected exactly one chain, found {len(chains)}"
    
    residues = list(chains[0].residues())
    assert len(residues) == 1, f"Expected exactly one residue, found {len(residues)}"
    
    # Get the single chain and residue
    old_chain = chains[0]
    old_residue = residues[0]
    
    # Determine the new residue name
    new_residue_name = rename_residue if rename_residue else old_residue.name
    
    # Validate mapping
    natoms = len(list(topology.atoms()))
    _check_renumber_mapping_dict(mapping, natoms)
    
    # Create reverse mapping (new index -> old index)
    rev_mapping = {v: k for k, v in mapping.items()}
    
    # Get all atoms in old order
    old_atoms = list(topology.atoms())
    
    # Create new topology with renumbered atoms
    new_topology = app.Topology()
    new_positions = []
    
    # Create the single chain and residue
    new_chain = new_topology.addChain(old_chain.id)
    new_residue = new_topology.addResidue(new_residue_name, new_chain, old_residue.id)
    
    # Add atoms in the new order
    new_atom_list = []
    for new_idx in range(natoms):
        old_idx = rev_mapping[new_idx]
        old_atom = old_atoms[old_idx]
        
        new_atom = new_topology.addAtom(
            old_atom.name,
            old_atom.element,
            new_residue,
            old_atom.id
        )
        new_atom_list.append(new_atom)
        vec = positions[old_idx]
        new_positions.append([vec.x, vec.y, vec.z])
    
    new_positions = np.array(new_positions) * 10
    
    # Add bonds in the new order
    for bond in topology.bonds():
        atom1_old_idx = old_atoms.index(bond[0])
        atom2_old_idx = old_atoms.index(bond[1])
        atom1_new_idx = mapping[atom1_old_idx]
        atom2_new_idx = mapping[atom2_old_idx]
        
        new_topology.addBond(new_atom_list[atom1_new_idx], new_atom_list[atom2_new_idx])
    
    # Set periodic box vectors if present
    if topology.getPeriodicBoxVectors() is not None:
        new_topology.setPeriodicBoxVectors(topology.getPeriodicBoxVectors())
    
    # Write the renumbered PDB file
    out = StringIO()
    app.PDBFile.writeFile(new_topology, new_positions, out, keepIds=True)
    out.seek(0)
    new_pdb_str = out.read()
    if output_pdb:
        with open(str(output_pdb), 'w') as f:
            f.write(new_pdb_str)
    return new_pdb_str


def rename_ff_xml(input_xml: os.PathLike, new_name: str, output_xml: os.PathLike = None):
    """
    Rename the residue in an OpenMM force field XML file.
    
    This function reads an OpenMM force field XML file, finds the residue name
    in the `<Residue>` tag, and replaces it with a new name. It also replaces
    all occurrences of the old residue name in type-related attributes throughout
    the XML file. The function assumes the XML file contains exactly one residue
    definition.
    
    Parameters
    ----------
    input_xml : os.PathLike
        Path to the input OpenMM force field XML file.
    new_name : str
        New name to assign to the residue. This will replace the `name` attribute
        in the `<Residue>` tag and all type attributes (e.g., "MOL-C1" -> "LIA-C1").
    output_xml : os.PathLike
        Path to the output XML file where the renamed force field will be saved.
    
    Raises
    ------
    AssertionError
        If the XML file does not contain exactly one `<Residue>` tag.
    
    Notes
    -----
    This function:
    
    1. Changes the `name` attribute in the `<Residue>` tag
    2. Replaces the old residue name prefix in all attributes that start with "type"
       (e.g., `type`, `type1`, `type2`, `type3`, `type4`) throughout the XML file
    3. The replacement pattern is: `{old_name}-{suffix}` -> `{new_name}-{suffix}`
       (e.g., "MOL-C1" -> "LIA-C1")
    
    The function preserves the XML structure and formatting as much as possible.
    
    Examples
    --------
    >>> rename_ff_xml('ligand.xml', 'LIA', 'ligand_renamed.xml')
    """
    # Parse the XML file
    if os.path.isfile(input_xml):
        tree = ET.parse(str(input_xml))
        root = tree.getroot()
    else:
        # input_xml is a string with XML content
        root = ET.fromstring(input_xml)
    
    # Find all Residue elements
    residues = root.findall('.//Residue')
    assert len(residues) == 1, f"Expected exactly one Residue tag, found {len(residues)}"
    
    # Get the single residue element and extract old name
    residue = residues[0]
    old_name = residue.get('name')
    
    # Replace the residue name attribute
    residue.set('name', new_name)
    
    # Find all elements in the XML tree and replace type attributes
    for elem in root.iter():
        # Check all attributes that start with "type"
        for attr_name in list(elem.attrib.keys()):
            if attr_name.startswith('type'):
                old_value = elem.get(attr_name)
                if old_value and old_value.startswith(f'{old_name}-'):
                    # Replace the prefix: old_name -> new_name
                    new_value = old_value.replace(f'{old_name}-', f'{new_name}-', 1)
                    elem.set(attr_name, new_value)
        
        # Also check the "name" attribute in AtomTypes/Type elements
        if elem.tag == 'Type':
            name_attr = elem.get('name')
            if name_attr and name_attr.startswith(f'{old_name}-'):
                # Replace the prefix: old_name -> new_name
                new_value = name_attr.replace(f'{old_name}-', f'{new_name}-', 1)
                elem.set('name', new_value)
    
    # Write the modified XML to output file with pretty printing
    return OpenmmXML.to_pretty_xmlfile(root, output_xml)


def generate_amber_mask(natomsA: int, natomsB: int, mapping: dict[int, int], alchemical_water_info: dict[str, list] = dict()):
    scA = [i for i in range(natomsA) if i not in mapping.keys()]
    scB = [i for i in range(natomsB) if i not in mapping.values()]
    res = {
        # "noshakemask": f"@1-{natomsA+natomsB}",
        "timask1": f"@1-{natomsA}",
        "timask2": f"@{natomsA+1}-{natomsA+natomsB}",
        "scmask1": "@{}".format(','.join(str(i+1) for i in scA)),
        "scmask2": "@{}".format(','.join(str(i+1+natomsA) for i in scB))
    }
    if alchemical_water_info:
        alchemical_water_mask = {
            # "noshakemask": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_water_oxygen'] + alchemical_water_info['alchemical_water_hydrogen'] + alchemical_water_info['alchemical_ions']),
            "timask1": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_water_oxygen'] + alchemical_water_info['alchemical_water_hydrogen']),
            "timask2": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_ions']),
            "scmask1": ','.join(str(x + 1) for x in alchemical_water_info['alchemical_water_hydrogen'])
        }
        for key in alchemical_water_mask:
            if not res[key]:
                res[key] = alchemical_water_mask[key]
            else:
                res[key] = f'{res[key]},{alchemical_water_mask[key]}'
    # add single quote mark
    for key in res:
        if res[key].startswith('@,'):
            res[key] = '@' + res[key][2:]
        res[key] = f"'{res[key]}'"
    return res


def preprocess_ligands(
    ligandA_sdf: os.PathLike, ligandA_pdb: os.PathLike, ligandA_ffxml: os.PathLike,
    ligandB_sdf: os.PathLike, ligandB_pdb: os.PathLike, ligandB_ffxml: os.PathLike,
    mapping: dict[int, int],
    ligandA_sdf_renumbered: os.PathLike, ligandA_pdb_renumbered: os.PathLike, ligandA_ffxml_renamed: os.PathLike,
    ligandB_sdf_renumbered: os.PathLike, ligandB_pdb_renumbered: os.PathLike, ligandB_ffxml_renamed: os.PathLike,
):
    """
    Renumber Ligands according to mapping and convert topology (.prmtop like) file to OpenMM XML
    """
    ligandA_mol = Chem.SDMolSupplier(str(ligandA_sdf), removeHs=False)[0]
    ligandB_mol = Chem.SDMolSupplier(str(ligandB_sdf), removeHs=False)[0]
    
    # Make renumbered topologies based on atom mapping
    # This make the order of common core atoms the same, which is a requirement of Amber hybrid topologies
    # generate renumber mapping
    ccA, ccB = [], []
    for k, v in mapping.items():
        ccA.append(k)
        ccB.append(v)
    renum_map_A = mk_renumber_mapping_from_cc(ccA, ligandA_mol.GetNumAtoms())
    renum_map_B = mk_renumber_mapping_from_cc(ccB, ligandB_mol.GetNumAtoms())

    # Renumber RDKit and save to sdf
    with Chem.SDWriter(str(ligandA_sdf_renumbered)) as w:
        ligandA_renum = renumber_rdkit_mol(ligandA_mol, renum_map_A)
        w.write(ligandA_renum)
    with Chem.SDWriter(str(ligandB_sdf_renumbered)) as w:
        ligandB_renum = renumber_rdkit_mol(ligandB_mol, renum_map_B)
        w.write(ligandB_renum)
    
    # Renumber parmed structure and save to pdb
    renumber_ligand_pdb(ligandA_pdb, renum_map_A, ligandA_pdb_renumbered, 'LIA')
    renumber_ligand_pdb(ligandB_pdb, renum_map_B, ligandB_pdb_renumbered, 'LIB')

    # Rename residue in ffxml
    rename_ff_xml(ligandA_ffxml, 'LIA', ligandA_ffxml_renamed)
    rename_ff_xml(ligandB_ffxml, 'LIB', ligandB_ffxml_renamed)

    # Atom mapping enforced: Amber requires mapped atoms to have distance < 0.1 Angstrom
    ligandA_renum = app.PDBFile(str(ligandA_pdb_renumbered))
    ligandB_renum = app.PDBFile(str(ligandB_pdb_renumbered))

    new_pos_B = np.array([[v.x, v.y, v.z] for v in ligandB_renum.positions]) * 10
    for i in range(len(mapping)):
        posA = ligandA_renum.positions[i]
        posB = ligandB_renum.positions[i]
        dist = math.sqrt((posA.x - posB.x) ** 2 + (posA.y - posB.y) ** 2 + (posA.z - posB.z) ** 2) * 10
        if dist > 0.1:
            logger.warning(f"Mapped atoms {i} have distance {dist:.3f} Angstroms (> 0.1), forcing alignment of these two atoms.")
            new_pos_B[i, 0] = posA.x * 10
            new_pos_B[i, 1] = posA.y * 10
            new_pos_B[i, 2] = posA.z * 10
    
    with open(ligandB_pdb_renumbered, 'w') as f:
        app.PDBFile.writeFile(ligandB_renum.topology, new_pos_B, f, keepIds=True)


def setup_ligand_rbfe_leg(
    ligandA_pdb: os.PathLike, ligandB_pdb: os.PathLike, # Ligand PDB/XML are processed, i.e. renumbered & renamed
    ligandA_xml: os.PathLike, ligandB_xml: os.PathLike,
    num_mapped_atoms: int,
    config: AmberFepSimulationConfig,
    wdir: os.PathLike,
    protein_pdb: os.PathLike | None = None,
    basename: str | None = None
):  
    # setup workding dir
    wdir = Path(wdir).expanduser().resolve()
    wdir.mkdir(exist_ok=True)
    basename = wdir.stem if not basename else basename

    # read pdb
    ligandA_pdb = app.PDBFile(str(ligandA_pdb))
    ligandB_pdb = app.PDBFile(str(ligandB_pdb))

    ligand_resnames = [
        list(ligandA_pdb.topology.residues())[0].name,
        list(ligandB_pdb.topology.residues())[0].name
    ]

    # charges
    ligandA_charge = compute_net_charge_from_openmm_system(
        app.ForceField(ligandA_xml).createSystem(ligandA_pdb.topology)
    )
    ligandB_charge = compute_net_charge_from_openmm_system(
        app.ForceField(ligandB_xml).createSystem(ligandB_pdb.topology)
    )

    # force field initialization
    ff = app.ForceField(*config.forcefields, ligandA_xml, ligandB_xml)

    # setup systems
    modeller = app.Modeller(app.Topology(), [])
    modeller.add(ligandA_pdb.topology, ligandA_pdb.positions)
    modeller.add(ligandB_pdb.topology, ligandB_pdb.positions)
    if protein_pdb:
        protein = app.PDBFile(str(protein_pdb))
        modeller.add(protein.topology, protein.positions)
    
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
            # This is for cases if A and B are stereo-isomers
            residueTemplates={res: res.name for res in modeller.topology.residues() if res.name in ligand_resnames}
        )
        fix_excess_charge(modeller, ligandB_charge)

    # generate masks
    num_atoms_A, num_atoms_B = len(list(ligandA_pdb.topology.atoms())), len(list(ligandB_pdb.topology.atoms()))
    mapping = {i:i for i in range(num_mapped_atoms)}
    mask = generate_amber_mask(num_atoms_A, num_atoms_B, mapping)

    # alchemical water
    d_charge = ligandB_charge - ligandA_charge
    alchem_waters, rst_settings = [], []
    if d_charge != 0:
        logger.info(f"Perturbation invoves charge change {int(ligandA_charge)} -> {int(ligandB_charge)}")
        if config.use_charge_change and (not config.gas_phase):
            scIndices = mask['scmask1'].strip("'")[1:].split(',') + mask['scmask2'].strip("'")[1:].split(',')
            scIndices = [int(x) - 1 for x in scIndices if x]
            alchem_water_info = do_co_alchemical_water(modeller, d_charge, scIndices)
            rst_settings = set_alchemical_water_restraints(modeller, scIndices, alchem_water_info)
            alchem_waters = [] if config.use_settle_for_alchemical_water else alchem_water_info['alchemical_water_residues']
            mask = generate_amber_mask(num_atoms_A, num_atoms_B, mapping, alchem_water_info)
        else:
            logger.warning("Charge change not enabled. Results are not trustworthy.")
    
    # again, this is for cases if A and B are stereo-isomers
    # because addSolvent will modify topology, we need to create again
    residueTemplates = {}
    for res in modeller.topology.residues():
        if res.name in ligand_resnames:
            residueTemplates[res] = res.name
        if all(name in residueTemplates for name in ligand_resnames):
            break
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, constraints=None, rigidWater=False, residueTemplates=residueTemplates)
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