import argparse
from typing import Sequence
from . import __version__


def parse_args(args: Sequence[str] | None = None):
    """
    EasyBFE command line argument parsers
    """
    parser = argparse.ArgumentParser(description="EasyBFE - a free, open-source software to setup and analyze binding free energy calculations")
    parser.add_argument(
        '--verbose',
        dest='verbose',
        help='Print as much information as possible. Useful for developers',
        action='store_true'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'EasyBFE - version {__version__}'
    )
    parser.add_argument(
        '-d', '--directory',
        dest='directory',
        help='Path to the EasyBFE project',
        default='.'
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")
    
    # Initialize
    init_parser = subparsers.add_parser("init", help="Initialize a project")
    
    # Add ligand
    add_ligands_parser = subparsers.add_parser('add_ligand', help='Add ligands')
    add_ligands_parser.add_argument(
        '-i', '--input',
        dest='input',
        required=True,
        nargs="+",
        help='Input sdf file'
    )
    add_ligands_parser.add_argument(
        '-n', '--name',
        dest='name',
        help='Name of the ligand. Only valid when the input sdf contains one molecule, otherwise will use the name in .sdf file'
    )
    add_ligands_parser.add_argument(
        '-p', '--protein_name',
        dest='protein_name',
        required=True,
        help='Name of the protein that the added ligand(s) belong to'
    )
    add_ligands_parser.add_argument(
        '-f', '--forcefield',
        dest='forcefield',
        default='bcc',
        help='Ligand forcefield. Valid options: "gaff", "gaff2" (acpype supported) and "openff-*" series (openff supported). When adding only one ligand, users can also specify a .prmtop/.top file as a customized forcefield for the ligand. Default is "gaff2"'
    )
    add_ligands_parser.add_argument(
        '-c', '--charge',
        dest='charge_method',
        help='Charge method. Valid options: "bcc" (default, am1-bcc model), "gas" (gasteiger model, inaccurate, never use it in production)'
    )
    add_ligands_parser.add_argument(
        '-m', '--num_workers',
        dest='num_workers',
        type=int,
        default=1,
        help='Number of process to run parametrization in parallel. Default is 1.'
    )
    add_ligands_parser.add_argument(
        '--no-parametrize',
        dest='parametrize',
        action='store_false',
        help='Disable parametrization for the ligands. Only useful in tests or you want to build your own force fields afterwards'
    )
    add_ligands_parser.add_argument(
        '--overwrite',
        dest='overwrite',
        action='store_true',
        help='Toggle to overwrite the existing ligand.'
    )
    add_ligands_parser.add_argument(
        '--expt',
        dest='expt',
        help='Experimental binding affinity of this ligand, in kcal/mol. Only valid when adding only one molecule, otherwise the binding affinity will be read from <dG.expt> or <affinity.expt> property.'
    )

    # Add protein
    add_protein_parser = subparsers.add_parser('add_protein', help='Add a protein')
    add_protein_parser.add_argument(
        '-i', '--input',
        dest='input',
        required=True,
        help='Path to the protein PDB file'
    )
    add_protein_parser.add_argument(
        '-n', '--name',
        dest='name',
        required=True,
        help='Name of this protein model'
    )
    add_protein_parser.add_argument(
        '--no-check-ff',
        dest='check_ff',
        action='store_false',
        help='Toggle to disable force field check.'
    )
    add_protein_parser.add_argument(
        '--overwrite',
        dest='overwrite',
        action='store_true',
        help='Toggle to overwrite the exisiting protein.'
    )

    # Add perturbation
    add_pert_parser = subparsers.add_parser('add_perturbation', help='Add perturbation')
    add_pert_parser.add_argument(
        '-p', '--protein_name',
        dest='protein_name',
        required=True
    )
    add_pert_parser.add_argument(
        '-l', '--list',
        dest='list',
        default=None,
        help='File contains list of perturabtions. In the file, each line contains two ligand names seperated by a whitespace'
    )
    add_pert_parser.add_argument(
        '--ligandA',
        dest='ligandA_name',
        help='Name of ligandA'
    )
    add_pert_parser.add_argument(
        '--ligandB',
        dest='ligandB_name',
        help='Name of ligandB'
    )
    add_pert_parser.add_argument(
        '-n', '--name',
        dest='pert_name',
        help='Name of the perturbation. Default will be `ligandA`~`ligandB`'
    )
    add_pert_parser.add_argument(
        '--config',
        dest='config',
        help='Configuration file',
        required=True
    )
    add_pert_parser.add_argument(
        '--mapping',
        dest='atom_mapping_method',
        default='kartograf',
        help='Method to perform mapping. Valid options: "lomap", "lazymcs", "kartograf"'
    )
    add_pert_parser.add_argument(
        '--mcs',
        dest='mcs',
        default=None,
        help="File (sdf) or SMILES or SMARTS that specify the common structure. Used only when mapping method is lazymcs"
    )
    add_pert_parser.add_argument(
        '-m', '--num_workers',
        dest='num_workers',
        type=int,
        default=1,
        help='Number of process to run parametrization in parallel. Default is 1.'
    )
    add_pert_parser.add_argument(
        '--submit',
        dest='submit',
        action='store_true',
        help='Toogle to submit the job with slurm (sbatch)'
    )
    add_pert_parser.add_argument(
        '--run-gas',
        dest='skip_gas',
        action='store_false',
        help='Toogle to also perform gas-phase simulation. Useful when you want to analyze solvation contribution. Only used when submit is True.'
    )
    add_pert_parser.add_argument(
        '--overwrite',
        dest='overwrite',
        action='store_true',
        help='Toggle to overwrite the exisiting perturbations.'
    )

    # Analysis
    ana_parser = subparsers.add_parser("analyze", help="Run analysis of Amber simulation results, including evaluate free energy with MBAR and post-processing trajectories")
    ana_parser.add_argument(
        '-p', '--protein_name',
        dest='protein_name',
        default="",
        help='Name of the protein'
    )
    ana_parser.add_argument(
        '-n', '--pert_name',
        dest='pert_name',
        default="",
        help='Name of the perturbation'
    )
    ana_parser.add_argument(
        '--skip_traj',
        dest='skip_traj',
        action="store_true",
        help="Skip analyzing simulation trajectories"
    )
    ana_parser.add_argument(
        '-m', '--num_workers',
        dest='num_workers',
        default=1,
        help="Number of processes to run the analysis in parallel",
        type=int
    )

    # Report
    report_parser = subparsers.add_parser("report", help='Report calculation results to a specified folder')
    report_parser.add_argument(
        '-o', '--output_dir',
        dest='save_dir',
        required=True,
        help='Output directory'
    )
    report_parser.add_argument(
        '-p', '--protein',
        dest='protein',
        default='',
        help='Report data belongs to a specific protein model. If not specified, all data will be reported.'
    )
    report_parser.add_argument(
        '-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Report verbose information for rbfe'
    )

    # constrain docking
    dock_parser = subparsers.add_parser("cdock", help='Run constrained docking workflow')
    dock_parser.add_argument('-i', '--input', required=True, dest='input', help='Input ligand: smiles string or sdf file')
    dock_parser.add_argument('-r', '--ref', required=True, dest='ref', help='Reference ligand (sdf) containing 3D structure constrained to')
    dock_parser.add_argument('-p', '--protein', required=True, dest='protein', help='Input protein pdb file')
    dock_parser.add_argument('-o', '--output', required=True, dest='output', help='Output directory')
    dock_parser.add_argument('-n', '--name', dest='name', help='Name of the ligand. Required if the input is a SMILES string.')
    dock_parser.add_argument('--prep_tool', dest='prep_tool', default='adfr', help='Tool used for prepare protein pdbqt. Vaild options: "adfr", "obabel"')
    
    args = parser.parse_args(args)
    return args


def main():
    from .amber import AmberRbfeProject

    args = parse_args()
    if args.verbose:
        print(args)
    
    if args.command == 'cdock':
        from .docking import VinaDocking
        docking = VinaDocking(args.protein, wdir=args.output, protein_prep_tool=args.prep_tool)
        docking.constr_dock(args.input, args.ref, args.name)
    else:
        project = AmberRbfeProject(args.directory, init=(args.command == 'init'))

    if args.command == 'analyze':
        if args.protein_name and args.pert_name:
            project.analyze_pert(args.protein_name, args.pert_name, args.skip_traj)
        else:
            project.analyze(num_workers=args.num_workers, skip_traj=args.skip_traj)
    elif args.command == 'add_ligand':
        # this handles where only one sdf is provided and that sdf contains multiple ligands
        inp = args.input[0] if len(args.input) == 1 else args.input
        project.add_ligands(
            inp,
            num_workers=args.num_workers,
            protein_name=args.protein_name,
            name=args.name,
            parametrize=args.parametrize,
            forcefield=args.forcefield,
            charge_method=args.charge_method,
            overwrite=args.overwrite,
            expt=args.expt
        )
    elif args.command == 'add_protein':
        project.add_protein(
            fpath=args.input,
            name=args.name,
            check_ff=args.check_ff,
            overwrite=args.overwrite
        )
    elif args.command == 'add_perturbation':
        kwargs = {
            'skip_gas': args.skip_gas,
            'submit': args.submit,
            'config': args.config,
            'overwrite': args.overwrite
        }
        if args.list is None:
            project.add_perturbation(
                args.ligandA_name, args.ligandB_name, args.protein_name, args.pert_name,
                **kwargs
            )
        else:
            project.add_perturbations(
                args.list, args.protein_name, args.num_workers,
                **kwargs
            )
    elif args.command == 'report':
        project.report(save_dir=args.save_dir, verbose=args.verbose, protein_name=args.protein)
        

if __name__ == '__main__':
    main()
