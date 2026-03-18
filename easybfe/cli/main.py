"""
EasyBFE command line interface using rich_click
"""
import json
import logging

import rich_click as click

from .. import __version__
from ..logging import setup_logging

from .abfe import abfe
from .ligand import ligand
from .md import md
from .rbfe import rbfe


@click.group()
@click.version_option(version=__version__, prog_name='EasyBFE')
@click.pass_context
def main(ctx):
    """
    EasyBFE - a free, open-source software to setup and analyze binding free energy calculations
    """
    # Configure logging once for the easybfe package so that
    # module-level loggers (logging.getLogger(__name__)) emit output.
    logger = logging.getLogger("easybfe")
    if not logger.handlers:
        setup_logging(log_file=None)
    ctx.ensure_object(dict)

main.add_command(abfe)
main.add_command(ligand)
main.add_command(md)
main.add_command(rbfe)

if __name__ == '__main__':
    main()

