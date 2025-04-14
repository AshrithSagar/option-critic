"""
oca/utils/cli.py \n
Command-line interface for loading and overriding configuration settings.
"""

import inspect
import io
import tokenize
from typing import Callable, Dict, Literal, Optional, Union, get_args, get_type_hints

import click
import yaml

from .config import (
    ConfigPlotsDefaults,
    ConfigPlotsProto,
    ConfigProto,
    ConfigRunDefaults,
    ConfigRunProto,
)
from .constants import __root__
from .plots import main as plots_main
from .run import main as run_main


def load_config(
    defaults: ConfigProto,
    config_path: Optional[str] = None,
    verbose: bool = False,
    **overrides,
) -> ConfigProto:
    config = defaults()
    # Override defaults with YAML file if provided
    if config_path:
        with open(config_path, "r") as file:
            config_dict: Dict = yaml.safe_load(file)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    # Override with command-line arguments
    for key, value in overrides.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    if verbose:
        print("Configuration:")
        for key in dir(config):
            if not key.startswith("_"):
                print(f"  {key}: {getattr(config, key)}")
    return config


def config_options(proto: ConfigProto) -> Callable:
    """Dynamically generate click options based on a Protocol's attributes"""

    def get_help_text() -> Dict[str, str]:
        """Extract comments from Protocol to use as help text."""
        source = inspect.getsource(proto)
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comments, prev_name, prev_tokval = {}, None, ""
        for toknum, tokval, _, _, _ in tokens:
            if toknum == tokenize.NAME:
                prev_tokval = tokval
            elif toknum == tokenize.OP and tokval == ":":
                prev_name = prev_tokval
            elif toknum == tokenize.COMMENT and prev_name:
                comments[prev_name] = tokval.strip("# ").strip()
                prev_name = None
        return comments

    def decorator(func):
        help_text = get_help_text()
        for name, type_hint in reversed(get_type_hints(proto).items()):
            # Map Python types to Click types
            if hasattr(type_hint, "__origin__"):
                args = get_args(type_hint)
                if type_hint.__origin__ is Literal:
                    click_type = click.Choice(args)
                elif type_hint.__origin__ is Union and type(None) in args:
                    literal_args = [arg for arg in args if arg is not type(None)]
                    click_type = (
                        click.Choice(get_args(literal_args[0]))
                        if len(literal_args) == 1
                        and getattr(literal_args[0], "__origin__", None) is Literal
                        else str
                    )
                else:
                    click_type = click.STRING
            else:
                click_type = {
                    str: click.STRING,
                    int: click.INT,
                    float: click.FLOAT,
                    bool: click.BOOL,
                    Optional[str]: click.STRING,
                    Optional[int]: click.INT,
                    Optional[float]: click.FLOAT,
                    Optional[bool]: click.BOOL,
                }.get(type_hint, str)

            # Add a click option for each attribute
            func = click.option(
                f"--{name.replace('_', '-')}",
                type=click_type,
                default=None,
                help=help_text.get(name),
            )(func)
        return func

    return decorator


def config_path(func):
    """Decorator to add config_path option to a function."""
    return click.option(
        "--config-path",
        type=click.Path(exists=True),
        default=None,
        help="Path to the YAML config file.",
    )(func)


@click.command()
@config_options(ConfigRunProto)
@config_path
def run(config_path: str, **kwargs):
    """Run an agent on an environment."""
    args = load_config(ConfigRunDefaults, config_path, verbose=True, **kwargs)
    run_main(args)


@click.command()
@config_options(ConfigPlotsProto)
@config_path
def plot(config_path: str, **kwargs):
    """Plotting utilities."""
    args = load_config(ConfigPlotsDefaults, config_path, verbose=True, **kwargs)
    plots_main(args)
