# flake8: noqa

"""
Monkey patch Python 3.7 breakpoint to use pdb or rpdb
or to pass through if we are not debugging

Useful only for experimental / test code where breakpoints
should be switched on and off fast.
"""
import sys
from loguru import logger
import slp.config as config

try:
    import ipdb as pdb
except:
    import pdb

# if config.REMOTE_DEBUGGING:
#     # import web_pdb as pdb
#     # Use socket-based rpdb because web_pdb cannot handle large structures
#     try:
#         import rpdb as pdb
#     except ImportError:
#         logger.warning("rpdb is not installed." "Remote debugging not available")
# else:
#     import pdb  # type: ignore


def set_trace():
    # if config.DEBUG:
    #     if config.REMOTE_DEBUGGING:
    #         pdb.set_trace(addr=config.DEBUG_ADDR, port=config.DEBUG_PORT)
    #     else:
    pdb.set_trace()


sys.breakpointhook = set_trace  # type: ignore


if __name__ == "__main__":
    print("hello")

    config.DEBUG = False

    # Should not stop execution here
    breakpoint()  # type: ignore

    print("world")

    config.DEBUG = True

    # Should stop execution here
    breakpoint()  # type: ignore

    print("!!!")
