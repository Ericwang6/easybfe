"""
EasyBFE command line interface using rich_click
"""
import json
import rich_click as click

from .. import __version__

from .abfe import abfe


@click.group()
@click.version_option(version=__version__, prog_name='EasyBFE')
@click.pass_context
def main(ctx):
    """
    EasyBFE - a free, open-source software to setup and analyze binding free energy calculations
    """
    pass

main.add_command(abfe)

if __name__ == '__main__':
    main()

