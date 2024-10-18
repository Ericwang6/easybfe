import argparse
from typing import Sequence


def parse_args(args: Sequence[str] | None = None):
    """
    EasyBFE command line argument parsers
    """
    parser = argparse.ArgumentParser(description="EasyBFE - an free, open-source package to easily setup relative binding free energies")
    parser.add_argument(
        '--verbose',
        dest='verbose',
        help='Print as much information as possible. Useful for developers',
        action='store_true'
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
        dest='name',
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
        '--interactive',
        dest='interactive',
        action='store_true',
        help='Use interactive mode'
    )
    
    args = parser.parse_args(args)
    return args


def main():
    from .amber import AmberRbfeProject

    args = parse_args()
    if args.verbose:
        print(args)
        
    project = AmberRbfeProject(args.directory, init=(args.command == 'init'))
    if args.command == 'analyze':
        if args.interactive:
            project.analyze(interactive=True)
        else:
            project.analyze_pert(args.protein_name, args.pert_name, args.skip_traj)
    elif args.command == 'add_ligand':
        project.add_ligands(
            args.input,
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
            'atom_mapping_method': args.atom_mapping_method,
            'config': args.config,
            'overwrite': args.overwrite
        }
        if args.atom_mapping_method == 'lazymcs':
            kwargs['atom_mapping_options'] = {'mcs': args.mcs}
        
        if args.list is None:
            project.add_perturbation(
                args.ligandA, args.ligandB, args.protein_name, args.pert_name,
                **kwargs
            )
        else:
            project.add_perturbations(
                args.list, args.protein_name, args.num_workers,
                **kwargs
            )
        

if __name__ == '__main__':
    main()