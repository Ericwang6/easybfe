"""
EasyBFE command line interface using rich_click
"""
import json
import rich_click as click

from .. import __version__
from ..config import *
from ..logging import setup_logging
from .utils import add_options_from_config


@click.group()
@click.option('--directory', '-d', help='Path to the EasyBFE project', default='.')
@click.option('--log', '-l', type=click.Path(), default=None, help='Path to log file')
@click.version_option(version=__version__, prog_name='EasyBFE')
@click.pass_context
def main(ctx, directory, log):
    """
    EasyBFE - a free, open-source software to setup and analyze binding free energy calculations
    """
    ctx.ensure_object(dict)
    ctx.obj['directory'] = directory
    setup_logging(log)


@main.command()
@click.pass_context
def init(ctx):
    """
    Initialize a project
    """
    from ..amber import AmberRbfeProject
    AmberRbfeProject(ctx.obj.get('directory'), init=True)


@main.command()
@click.option('--name', '-n', type=str, help='Name of the ligand (single molecule)')
@click.option('--protein-name', '-p', type=str, help='Name of the protein (defaults to project default if not specified)')
@click.option('--forcefield', '-f', type=str, default='gaff2', help='Force field: "gaff", "gaff2", "openff-*", or path to .prmtop/.top file. Default: "gaff2"')
@click.option('--charge-method', '-c', type=str, default='bcc', help='Charge method: "bcc" (AM1-BCC, default) or "gas" (Gasteiger, debug only)')
@click.option('--num-workers', '-m', type=int, default=-1, help='Number of parallel workers. Default: -1, which means as many processors as possible')
@click.option('--overwrite', is_flag=True, help='Overwrite existing ligands')
@click.option('--expt', type=float, help='Experimental binding affinity in kcal/mol (single molecule) or read from <dG.expt>/<affinity.expt> property')
@click.option('--smiles-col', type=str, help='Column name for SMILES when input is CSV')
@click.option('--name-col', type=str, help='Column name for ligand names when input is CSV')
@click.argument('input', nargs=-1, required=True, type=str, help="Input ligand file(s) (SDF, MOL, MOL2) or CSV file with SMILES column. Can specify multiple files.")
@click.pass_context
def add_ligand(ctx, name, protein_name, forcefield, charge_method, num_workers, overwrite, expt, smiles_col, name_col, input):
    """Add ligands to the project
    """
    from ..amber import AmberRbfeProject
    project = AmberRbfeProject(ctx.obj.get('directory'), num_workers=num_workers)
    inp = input[0] if len(input) == 1 else list(input)
    kwargs = {'forcefield': forcefield, 'charge_method': charge_method, 'overwrite': overwrite}
    if protein_name:
        kwargs['protein_name'] = protein_name
    if name and len(input) == 1:
        kwargs['name'] = name
    if expt is not None:
        kwargs['expt'] = expt
    if smiles_col:
        kwargs['smiles_col'] = smiles_col
    if name_col:
        kwargs['name_col'] = name_col
    project.add_ligands(inp, **kwargs)


@main.command()
@click.option('--name', '-n', type=str, required=False, help='Name of this protein model. If not provided, stem of the file will be used.')
@click.option('--overwrite', is_flag=True, help='Toggle to overwrite the existing protein.')
@click.option('--prepare/--no-prepare', default=True, help='Whether to run the preparation workflow')
@click.option('--prepare-config', '-c', type=click.Path(exists=True), help='Path to preparation config file.')
@add_options_from_config(ProteinPrepareConfig)
@click.argument('input', type=click.Path(exists=True), required=True, help="Path to the protein PDB file")
@click.pass_context
def add_protein(ctx, input, name, overwrite, prepare, prepare_config, **kwargs):
    """Add a protein to the project
    """
    from ..amber import AmberRbfeProject
    project = AmberRbfeProject(ctx.obj.get('directory'))
    if prepare:
        cfg_dict = read_file(prepare_config) if prepare_config else {}
        cfg_dict = cfg_dict['protein_prepare'] if 'protein_prepare' in cfg_dict else cfg_dict
        cfg = load_config(ProteinPrepareConfig, cfg_dict, kwargs)
    else:
        cfg = None

    project.add_protein(fpath=input, name=name, overwrite=overwrite, do_prepare=prepare, prepare_config=cfg)


@main.command()
@click.option('--protein-name', '-p', type=str, help='Name of the protein')
@click.option('--ligandA', type=str, help='Name of ligand A (required if --list is not provided)')
@click.option('--ligandB', type=str, help='Name of ligand B (required if --list is not provided)')
@click.option('--pert-name', '-n', type=str, help='Name of the perturbation (only used when --list is not provided)')
@click.option('--list', '-l', 'perturbation_list', type=click.Path(exists=True, file_okay=True, dir_okay=False), 
    help='File contains list of perturbations. In the file, each line contains two ligand names separated by a whitespace'
)
@click.option('--config', '-c', required=True, help='Configuration file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    '--num_workers', '-m', type=int, default=-1,
    help='Number of process to run parametrization in parallel. Default is -1.'
)
@click.option('--overwrite', is_flag=True, help='Toggle to overwrite the existing perturbation directories.')
@click.pass_context
def add_perturbation(ctx, protein_name, ligandA, ligandB, pert_name, perturbation_list, config, num_workers, submit, run_gas, overwrite, ):
    """
    Add perturbation(s) to the project
    """
    from ..amber import AmberRbfeProject

    project = AmberRbfeProject(ctx.obj.get('directory'), num_workers=num_workers)
    
    if perturbation_list is None:
        if ligandA is None or ligandB is None:
            raise click.BadParameter("--ligandA and --ligandB are required when --list is not provided")
        
        project.add_perturbation(ligandA_name=ligand_a,
            ligandB_name=ligand_b,
            protein_name=protein_name,
            pert_name=pert_name,
            **kwargs
        )
    else:
        kwargs['protein_name'] = protein_name
        project.add_perturbations(
            config=config_obj,
            perturbations=perturbation_list,
            num_workers=num_workers,
            **kwargs
        )


@main.command()
@click.option(
    '--protein_name', '-p',
    type=str,
    default='',
    help='Name of the protein'
)
@click.option(
    '--pert_name', '-n',
    type=str,
    default='',
    help='Name of the perturbation'
)
@click.option(
    '--skip_traj',
    is_flag=True,
    help='Skip analyzing simulation trajectories'
)
@click.option(
    '--num_workers', '-m',
    type=int,
    default=1,
    help='Number of processes to run the analysis in parallel'
)
@click.pass_context
def analyze(ctx, protein_name, pert_name, skip_traj, num_workers):
    """
    Run analysis of Amber simulation results, including evaluate free energy with MBAR and post-processing trajectories
    """
    project = ctx.obj.get('project')
    
    if protein_name and pert_name:
        project.analyze_pert(protein_name, pert_name, skip_traj)
    else:
        project.analyze(num_workers=num_workers, skip_traj=skip_traj)


@main.command()
@click.option(
    '--output_dir', '-o',
    type=str,
    required=True,
    help='Output directory'
)
@click.option(
    '--protein', '-p',
    type=str,
    default='',
    help='Report data belongs to a specific protein model. If not specified, all data will be reported.'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Report verbose information for rbfe'
)
@click.pass_context
def report(ctx, output_dir, protein, verbose):
    """
    Report calculation results to a specified folder
    """
    project = ctx.obj.get('project')
    project.report(save_dir=output_dir, verbose=verbose, protein_name=protein)


@main.command()
@click.option(
    '--protein', '-p',
    type=str,
    required=True,
    help='Name of the protein'
)
@click.option(
    '--ligand', '-l',
    type=str,
    default='',
    help='Name of the ligand'
)
@click.option(
    '--task_name', '-t',
    type=str,
    required=True,
    help='Name of the task'
)
@click.option(
    '--config', '-c',
    type=str,
    required=True,
    help='Configuration file'
)
@click.option(
    '--ligand-only',
    is_flag=True,
    help='Only simulate ligand'
)
@click.option(
    '--submit',
    is_flag=True,
    help='Submit the job'
)
@click.option(
    '--overwrite',
    is_flag=True,
    help='Overwrite existing files'
)
@click.pass_context
def md(ctx, protein, ligand, task_name, config, ligand_only, submit, overwrite):
    """
    Run plain MD simulation
    """
    project = ctx.obj.get('project')
    project.run_plain_md(
        protein, ligand, task_name, config,
        ligand_only, submit, overwrite
    )


@main.command()
@click.option(
    '--task_name', '-t',
    type=str,
    default='',
    help='Name of the plain MD task'
)
@click.option(
    '--task_dir',
    type=str,
    default='',
    help='Path to the plain MD task directory'
)
@click.option(
    '--config', '-c',
    type=str,
    default='',
    help='Path to the configuration file for analysis'
)
@click.pass_context
def analyze_md(ctx, task_name, task_dir, config):
    """
    Analyze results for plain MD
    """
    project = ctx.obj.get('project')
    project.analyze_md(task_name, task_dir, config)


@main.command()
@click.option(
    '--input', '-i',
    type=str,
    required=True,
    help='Input ligand: smiles string or sdf file'
)
@click.option(
    '--ref', '-r',
    type=str,
    required=True,
    help='Reference ligand (sdf) containing 3D structure constrained to'
)
@click.option(
    '--protein', '-p',
    type=str,
    required=True,
    help='Input protein pdb file'
)
@click.option(
    '--output', '-o',
    type=str,
    required=True,
    help='Output directory'
)
@click.option(
    '--name', '-n',
    type=str,
    help='Name of the ligand. Required if the input is a SMILES string.'
)
@click.option(
    '--prep_tool',
    type=str,
    default='adfr',
    help='Tool used for prepare protein pdbqt. Valid options: "adfr", "obabel"'
)
def cdock(input, ref, protein, output, name, prep_tool):
    """
    Run constrained docking workflow
    """
    from ..docking import VinaDocking
    
    docking = VinaDocking(protein, wdir=output, protein_prep_tool=prep_tool)
    docking.constr_dock(input, ref, name)


if __name__ == '__main__':
    main()

