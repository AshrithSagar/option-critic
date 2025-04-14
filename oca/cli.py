"""
oca/cli.py
Main command-line interface
"""

import click

from .utils.cli import evaluate, plot, run
from .utils.constants import __version__


@click.group()
@click.version_option(__version__)
def oca():
    """Option-Critic Architecture CLI"""
    pass


oca.add_command(run)
oca.add_command(evaluate)
oca.add_command(plot)
