# flake8: noqa

"""
Monkey patch Python 3.7 breakpoint to use pdb or rpdb
or to pass through if we are not debugging

Useful only for experimental / test code where breakpoints
should be switched on and off fast.
"""
import sys
from loguru import logger

try:
    import ipdb as pdb
except:
    import pdb


def set_trace():
    pdb.set_trace()


sys.breakpointhook = set_trace  # type: ignore
