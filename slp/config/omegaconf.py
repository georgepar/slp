import argparse
import pathlib
import sys
from collections import defaultdict
from typing import IO, Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def _nest(
    d: Dict[str, Any], separator: str = ".", include_none: bool = False
) -> Optional[Dict[str, Any]]:
    """_nest Recursive function to nest a dictionary on keys with . (dots)

    Parse documentation into a hierarchical dict. Keys should be separated by dots (e.g. "model.hidden") to go down into the hierarchy

    Args:
        d (Dict[str, Any]): dictionary containing flat config values
        separator (str): Separator to nest dictionary
        include_none (bool): If true includes none values in final dict

    Returns:
        Dict[str, Any]: Hierarchical config dictionary

    Examples:
        >>> _nest({{"model.hidden": 20, "optimizer.lr": 1e-3}})
        {"model": {"hidden": 20}, "optimizer": {"lr": 1e-3}}
    """
    nested: Dict[str, Any] = defaultdict(dict)

    for key, val in d.items():
        if "." in key:
            splitkeys = key.split(".")
            inner = _nest({".".join(splitkeys[1:]): val})

            if inner is not None:
                nested[splitkeys[0]].update(inner)
        else:
            if val is not None:
                nested[key] = val

            if val is None and include_none:
                nested[key] = val

    return dict(nested) if nested else None


class OmegaConfExtended(OmegaConf):
    """OmegaConfExtended Extended OmegaConf class, to include argparse style CLI arguments

    Unfortunately the original authors are not interested into providing integration with argparse
    (https://github.com/omry/omegaconf/issues/569), so we have to get by with this extension
    """

    @staticmethod
    def from_yaml(
        file_: Union[str, pathlib.Path, IO[Any]]
    ) -> Union[DictConfig, ListConfig]:
        """Alias for OmegaConf.load
        OmegaConf.from_yaml got removed at some point. Bring it back

        Args:
            file_ (Union[str, pathlib.Path, IO[Any]]): file to load or file descriptor

        Returns:
            Union[DictConfig, ListConfig]: The loaded configuration

        """
        return OmegaConfExtended.load(file_)

    @staticmethod
    def from_argparse(
        parser: argparse.ArgumentParser,
        args: Optional[List[str]] = None,
        include_none: bool = False,
    ) -> Tuple[DictConfig, DictConfig]:
        """from_argparse Static method to convert argparse arguments into OmegaConf DictConfig objects

        We parse the command line arguments and separate the user provided values and the default values.
        This is useful for merging with a config file.

        Args:
            parser (argparse.ArgumentParser): Parser for argparse arguments
            args (Optional[List[str]]): Optional input sys.argv style args. Useful for testing.
                Use this only for testing. By default it uses sys.argv[1:]
        Returns:
            Tuple[omegaconf.DictConfig, omegaconf.DictConfig]: (user provided cli args, default cli args) as a tuple of omegaconf.DictConfigs

        Examples:
            >>> import argparse
            >>> from slp.config.omegaconf import OmegaConfExtended
            >>> parser = argparse.ArgumentParser("My cool model")
            >>> parser.add_argument("--hidden", dest="model.hidden", type=int, default=20)
            >>> user_provided_args, default_args = OmegaConfExtended.from_argparse(parser, args=["--hidden", "100"])
            >>> user_provided_args
            {'model': {'hidden': 100}}
            >>> default_args
            {}
            >>> user_provided_args, default_args = OmegaConfExtended.from_argparse(parser)
            >>> user_provided_args
            {}
            >>> default_args
            {'model': {'hidden': 20}}
        """
        dest_to_arg = {v.dest: k for k, v in parser._option_string_actions.items()}

        all_args = vars(parser.parse_args(args=args))
        provided_args = {}
        default_args = {}

        for k, v in all_args.items():
            if dest_to_arg[k] in sys.argv:
                provided_args[k] = v
            else:
                default_args[k] = v

        provided = OmegaConf.create(_nest(provided_args, include_none=include_none))
        defaults = OmegaConf.create(_nest(default_args, include_none=include_none))

        return provided, defaults
