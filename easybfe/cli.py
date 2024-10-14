import argparse
from typing import Sequence


def parse_args(args: Sequence[str] | None = None):
    """
    EasyBFE command line argument parsers
    """
    parser = argparse.ArgumentParser(description="EasyBFE - an free, open-source package to easily setup relative binding free energies")
    parser.add_argument(
        '-d', '--directory',
        dest='directory',
        help='Path to the EasyBFE project',
        default='.'
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")
    init_parser = subparsers.add_parser("init", help="Initialize a project")
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
    project = AmberRbfeProject(args.directory, init=(args.command == 'init'))
    if args.command == 'analyze':
        if args.interactive:
            project.analyze_interactive()
        else:
            project.analyze_pert(args.protein_name, args.pert_name, args.skip_traj)
        

if __name__ == '__main__':
    main()