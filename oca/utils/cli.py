"""
utils/cli.py \n
Command-line interface for loading and overriding configuration settings.
"""

import inspect
import io
import sys
import tokenize
from typing import Dict, Optional, Protocol, get_type_hints

import click
import yaml

from .config import ConfigDefaults, ConfigProto


def get_config(config_path: Optional[str] = None, **overrides) -> ConfigProto:
    config = ConfigDefaults()
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
    return config


def options_from_proto(proto: Protocol):
    """Dynamically generate click options based on a Protocol's attributes"""

    def get_help_text() -> Dict[str, str]:
        """Extract comments from the Protocol to use as help text."""
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
            click_type = {
                str: str,
                int: int,
                float: float,
                bool: bool,
                Optional[str]: str,
                Optional[int]: int,
                Optional[float]: float,
                Optional[bool]: bool,
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


def load_config(verbose: bool = False) -> ConfigProto:
    """Handles CLI arguments and returns the final configuration."""

    @click.command(help="Option Critic Architecture | PyTorch")
    @options_from_proto(ConfigProto)
    @click.option(
        "--config-path",
        type=click.Path(exists=True),
        default=None,
        help="Path to the YAML config file.",
    )
    def cli(config_path, **kwargs):
        config = get_config(config_path, **kwargs)
        if verbose:
            print("Configuration:")
            for key in dir(config):
                if not key.startswith("_"):
                    print(f"  {key}: {getattr(config, key)}")
        return config

    # Early exit if --help
    standalone_mode = "--help" in sys.argv
    return cli.main(standalone_mode=standalone_mode)
