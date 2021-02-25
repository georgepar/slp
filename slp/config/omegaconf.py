import argparse
import sys

from omegaconf import OmegaConf

from collections import defaultdict


def _nest(d):
    nested = defaultdict(dict)
    for key, val in d.items():
        if "." in key:
            splitkeys = key.split(".")
            inner = _nest({".".join(splitkeys[1:]): val})
            if inner is not None:
                nested[splitkeys[0]].update(inner)
        else:
            if val is not None:
                nested[key] = val
    return dict(nested) if nested else None


class OmegaConfExtended(OmegaConf):
    @staticmethod
    def from_argparse(parser: argparse.ArgumentParser):
        dest_to_arg = {v.dest: k for k, v in parser._option_string_actions.items()}

        all_args = vars(parser.parse_args())
        provided_args = {}
        default_args = {}
        for k, v in all_args.items():
            if dest_to_arg[k] in sys.argv:
                provided_args[k] = v
            else:
                default_args[k] = v

        provided = OmegaConf.create(_nest(provided_args))
        defaults = OmegaConf.create(_nest(default_args))

        return provided, defaults