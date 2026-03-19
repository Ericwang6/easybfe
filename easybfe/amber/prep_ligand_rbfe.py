import json
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
from ..config import AmberFepSimulationConfig, AmberWtSettings
from ..parallel import run_func_parallel
from .workflow import Step, Workflow, create_script_for_workflows
from ..smff.utils import OpenmmXML
from ..core import Ligand, Protein
from ..mapping import load_mapper
from ..config.amber.rbfe import AmberLigandRbfeConfig, AtomMappingConfig


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
    if isinstance(input_pdb, (str, Path)) and os.path.isfile(input_pdb):
        pdb = app.PDBFile(str(input_pdb))
    else:
        inp = StringIO(input_pdb)
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


def preprocess_ligands(ligandA: Ligand, ligandB: Ligand, mapping: dict[int, int]) -> tuple[Ligand, Ligand]:
    """
    Renumber Ligands according to mapping and convert topology (.prmtop like) file to OpenMM XML
    """
    ligandA_mol = ligandA.get_rdmol()
    ligandB_mol = ligandB.get_rdmol()
    
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
    ligandA_mol_renum = renumber_rdkit_mol(ligandA_mol, renum_map_A)
    ligandB_mol_renum = renumber_rdkit_mol(ligandB_mol, renum_map_B)
    
    # Renumber parmed structure and save to pdb
    ligandA_pdb_processed = renumber_ligand_pdb(ligandA.auxiliary_files['pdb'], renum_map_A, rename_residue='LIA')
    ligandB_pdb_processed = renumber_ligand_pdb(ligandB.auxiliary_files['pdb'], renum_map_B, rename_residue='LIB')

    # Rename residue in ffxml
    ligandA_xml_processed = rename_ff_xml(ligandA.auxiliary_files['xml'], 'LIA')
    ligandB_xml_processed = rename_ff_xml(ligandB.auxiliary_files['xml'], 'LIB')

    # Atom mapping enforced: Amber requires mapped atoms to have distance < 0.1 Angstrom
    buf = StringIO()
    buf.write(ligandA_pdb_processed)
    buf.seek(0)
    ligandA_pdb_renum = app.PDBFile(buf)
    buf.seek(0)
    buf.truncate(0)
    
    buf.write(ligandB_pdb_processed)
    buf.seek(0)
    ligandB_pdb_renum = app.PDBFile(buf)
    buf.seek(0)
    buf.truncate(0)

    new_pos_B = np.array([[v.x, v.y, v.z] for v in ligandB_pdb_renum.positions]) * 10
    for i in range(len(mapping)):
        posA = ligandA_pdb_renum.positions[i]
        posB = ligandB_pdb_renum.positions[i]
        dist = math.sqrt((posA.x - posB.x) ** 2 + (posA.y - posB.y) ** 2 + (posA.z - posB.z) ** 2) * 10
        if dist > 0.1:
            logger.warning(f"Mapped atoms {i} have distance {dist:.3f} Angstroms (> 0.1), forcing alignment of these two atoms.")
            new_pos_B[i, 0] = posA.x * 10
            new_pos_B[i, 1] = posA.y * 10
            new_pos_B[i, 2] = posA.z * 10
    
    app.PDBFile.writeFile(ligandB_pdb_renum.topology, new_pos_B, buf, keepIds=True)
    buf.seek(0)
    ligandB_pdb_processed = buf.read()
    
    ligandA_data = ligandA.model_dump()
    ligandA_data['mol_block'] = Chem.MolToMolBlock(ligandA_mol_renum)
    ligandA_data['auxiliary_files']['pdb'] = ligandA_pdb_processed
    ligandA_data['auxiliary_files']['xml'] = ligandA_xml_processed

    ligandB_data = ligandB.model_dump()
    ligandB_data['mol_block'] = Chem.MolToMolBlock(ligandB_mol_renum)
    ligandB_data['auxiliary_files']['pdb'] = ligandB_pdb_processed
    ligandB_data['auxiliary_files']['xml'] = ligandB_xml_processed

    return (Ligand.model_validate(ligandA_data), Ligand.model_validate(ligandB_data))


def setup_ligand_rbfe_leg(
    ligandA: Ligand, ligandB: Ligand, # Ligand are processed, i.e. renumbered & renamed
    num_mapped_atoms: int,
    protein: Protein | None,
    config: AmberFepSimulationConfig,
    wdir: os.PathLike,
    basename: str | None = None
):  
    # setup workding dir
    wdir = Path(wdir).expanduser().resolve()
    wdir.mkdir(exist_ok=True)
    basename = wdir.stem if not basename else basename

    # dump ligandA and B data
    ligandA.name = 'ligandA_processed'
    ligandB.name = 'ligandB_processed'
    ligandA.dump(wdir)
    ligandB.dump(wdir)

    # read pdb
    ligandA_pdb = app.PDBFile(str(wdir / f'{ligandA.name}.pdb'))
    ligandB_pdb = app.PDBFile(str(wdir / f'{ligandB.name}.pdb'))

    ligand_resnames = [
        list(ligandA_pdb.topology.residues())[0].name,
        list(ligandB_pdb.topology.residues())[0].name
    ]

    # charges
    ligandA_charge = compute_net_charge_from_openmm_system(
        app.ForceField(str(wdir / f'{ligandA.name}.xml')).createSystem(ligandA_pdb.topology)
    )
    ligandB_charge = compute_net_charge_from_openmm_system(
        app.ForceField(str(wdir / f'{ligandB.name}.xml')).createSystem(ligandB_pdb.topology)
    )

    # force field initialization
    ff = app.ForceField(*config.forcefields, str(wdir / f'{ligandA.name}.xml'), str(wdir / f'{ligandB.name}.xml'))

    # setup systems
    modeller = app.Modeller(app.Topology(), [])
    modeller.add(ligandA_pdb.topology, ligandA_pdb.positions)
    modeller.add(ligandB_pdb.topology, ligandB_pdb.positions)
    if protein:
        protein_openmm = protein.to_openmm()
        modeller.add(protein_openmm.topology, protein_openmm.positions)
    
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

    # alchemical water / co-alchemical ion for charge-change perturbations
    d_charge = ligandB_charge - ligandA_charge
    alchem_waters, rst_settings = [], []
    if d_charge != 0:
        logger.info(f"Perturbation involves charge change {int(ligandA_charge)} -> {int(ligandB_charge)}")
        if config.use_charge_change and (not config.gas_phase):
            scIndices = mask['scmask1'].strip("'")[1:].split(',') + mask['scmask2'].strip("'")[1:].split(',')
            scIndices = [int(x) - 1 for x in scIndices if x]
            coion_info = create_alchemical_ions(
                modeller, int(ligandA_charge), int(ligandB_charge), scIndices,
                method=config.charge_change_method,
            )
            if config.add_restraint_for_alchem_water:
                rst_settings = set_alchemical_water_restraints(modeller, scIndices, coion_info)
            alchem_waters = [] if config.use_settle_for_alchemical_water else coion_info['alchemical_water_residues']
            mask = generate_amber_mask(num_atoms_A, num_atoms_B, mapping, coion_info)
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
    workflows = []
    for n, clambda in enumerate(config.lambdas):
        steps = []
        for step_template in config.workflow:
            update = mask.copy()
            update.update({
                "clambda": clambda,
                'ntwx': 10 * step_template.cntrl.ntwx if (config.reduce_storage and 0 < n < len(config.lambdas) - 1) else step_template.cntrl.ntwx,
            })
            base = step_template.cntrl.model_dump()
            base.update(update)
            step_lambda_cntrl = step_template.cntrl.__class__.model_validate(base)
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

    create_script_for_workflows(workflows, wdir, config.num_procs)

    return True


def setup_ligand_rbfe(
    ligandA: Ligand, ligandB: Ligand, 
    mapping: dict[int, int] | AtomMappingConfig | None, 
    protein: Protein | None,
    leg_configs: dict[str, AmberFepSimulationConfig],
    output_dir: os.PathLike,
    reuse_mapping: bool = True
):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(exist_ok=True)
    
    atom_mapping_json = output_dir / 'atom_mapping.json'
    if reuse_mapping and atom_mapping_json.is_file():
        with open(atom_mapping_json) as f:
            mapping = {int(k): int(v) for k, v in json.load(f).items()}
    else:
        if isinstance(mapping, dict):
            with open(atom_mapping_json, 'w') as f:
                json.dump(mapping, f, indent=4)
        else:
            mapping_cfg = AtomMappingConfig() if mapping is None else mapping
            mapper = load_mapper(mapping_cfg.algorithm, **mapping_cfg.options)
            mapping = mapper.run(ligandA, ligandB, output_dir)
    
    ligandA_processed, ligandB_processed = preprocess_ligands(ligandA, ligandB, mapping)
    
    for leg, config in leg_configs.items():
        if 'gas' in leg and (not config.gas_phase):
            config.gas_phase = True
            logger.warning("set gas_phase=True forcibly in gas phase simulation config")
        leg_protein = protein if 'complex' in leg else None
        setup_ligand_rbfe_leg(ligandA_processed, ligandB_processed, len(mapping), leg_protein, config, output_dir/leg, basename='system')


def _pair_subdir_name(p0: os.PathLike, p1: os.PathLike) -> str:
    """Build ``pair[0]~pair[1]`` output segment; flatten path separators for one directory name."""
    s0 = Path(p0).as_posix().replace("/", "_")
    s1 = Path(p1).as_posix().replace("/", "_")
    return f"{s0}~{s1}"


def _resolve_ligand_directory(ligand_base: Path | None, component: os.PathLike) -> Path:
    """Resolve a ligand directory under optional ``ligand_base`` or as an absolute path."""
    comp = Path(component)
    if ligand_base is not None:
        return (Path(ligand_base).expanduser().resolve() / comp).resolve()
    return comp.expanduser().resolve()


def _setup_ligand_rbfe_one(job: tuple) -> None:
    """Load ligands from directories and run :func:`setup_ligand_rbfe` (batch / parallel entry point).

    Parameters
    ----------
    job : tuple
        ``(ligand_a_dir, ligand_b_dir, mapping, protein, leg_configs, output_dir)``.
    """
    ligand_a_dir, ligand_b_dir, mapping, protein, leg_configs, output_dir = job
    ligandA = Ligand.from_directory(ligand_a_dir)
    ligandB = Ligand.from_directory(ligand_b_dir)
    setup_ligand_rbfe(
        ligandA=ligandA,
        ligandB=ligandB,
        mapping=mapping,
        protein=protein,
        leg_configs=leg_configs,
        output_dir=output_dir,
    )


def setup_ligand_rbfe_from_config(
    config: AmberLigandRbfeConfig,
    num_procs: int | None = None,
) -> None:
    """Run RBFE setup from an :class:`AmberLigandRbfeConfig`.

    **Ligand directories**

    If :attr:`~AmberLigandRbfeConfig.ligand_base` is set, single-pair mode uses
    ``ligand_base / ligandA`` and ``ligand_base / ligandB``; batch mode (non-empty
    :attr:`~AmberLigandRbfeConfig.ligand_pairs`) uses ``ligand_base / pair[0]`` and
    ``ligand_base / pair[1]`` for each pair. If ``ligand_base`` is not set,
    :attr:`~AmberLigandRbfeConfig.ligandA` / ``ligandB`` (or each pair entry) are
    treated as full ligand directory paths.

    **Output**

    Batch mode requires :attr:`~AmberLigandRbfeConfig.output_base`; each run writes
    under ``output_base / "<pair0>~<pair1>"``. Single-pair mode uses
    ``output_base / "{ligandA.name}~{ligandB.name}"`` when ``output_base`` is set,
    otherwise :attr:`~AmberLigandRbfeConfig.output_dir` (required in that case).
    """
    assert config.protein is not None, "AmberLigandRbfeConfig.protein must be set"

    leg_configs = {
        "complex": config.complex,
        "solvent": config.solvent,
    }
    if config.gas is not None:
        leg_configs["gas"] = config.gas

    protein = Protein.from_pdb(config.protein, name=config.protein.stem)

    use_pairs = config.ligand_pairs is not None and len(config.ligand_pairs) > 0
    lig_base = Path(config.ligand_base).expanduser().resolve() if config.ligand_base is not None else None

    if use_pairs:
        if config.output_base is None:
            raise ValueError(
                "AmberLigandRbfeConfig.output_base is required for batch mode (non-empty ligand_pairs)"
            )
        if config.ligandA is not None or config.ligandB is not None:
            logger.info("ligand_pairs batch mode: ignoring ligandA and ligandB for ligand paths")
        out_base = Path(config.output_base).expanduser().resolve()
        out_base.mkdir(parents=True, exist_ok=True)
        nprocs = num_procs if num_procs is not None else -1
        args_list = [
            (
                _resolve_ligand_directory(lig_base, p0),
                _resolve_ligand_directory(lig_base, p1),
                config.atom_mapping,
                protein,
                leg_configs,
                out_base / _pair_subdir_name(p0, p1),
            )
            for p0, p1 in config.ligand_pairs
        ]
        run_func_parallel(
            _setup_ligand_rbfe_one,
            args_list,
            nprocs=nprocs,
            desc="setup_ligand_rbfe",
        )
        return

    if config.ligandA is None or config.ligandB is None:
        raise ValueError(
            "AmberLigandRbfeConfig.ligandA and ligandB are required when ligand_pairs is not set"
        )

    if config.output_base is not None:
        run_out = (
            Path(config.output_base).expanduser().resolve()
            / f"{Path(config.ligandA).name}~{Path(config.ligandB).name}"
        )
    elif config.output_dir is not None:
        run_out = Path(config.output_dir).expanduser().resolve()
    else:
        raise ValueError(
            "Set output_base or output_dir for single-pair RBFE setup "
            "(output_dir is required when output_base is not set)"
        )

    run_out.mkdir(parents=True, exist_ok=True)
    _setup_ligand_rbfe_one(
        (
            _resolve_ligand_directory(lig_base, config.ligandA),
            _resolve_ligand_directory(lig_base, config.ligandB),
            config.atom_mapping,
            protein,
            leg_configs,
            run_out,
        )
    )