import sys

import slp.util.log as log

LOGGER = log.getLogger('slp')

try:
    #import web_pdb as pdb
    # Use socket-based rpdb because web_pdb cannot handle large structures
    import rpdb
    pdb = rpdb.Rpdb(addr='0.0.0.0')
except ImportError:
    LOGGER.warning('rpdb is not installed.'
                   'Remote debugging not available')
    import pdb

import slp.config as config

def set_trace():
    if config.DEBUG:
        pdb.set_trace()

sys.breakpointhook = set_trace


if __name__ == '__main__':
    print('hello')
    # Should not stop execution here
    config.DEBUG = False

    breakpoint()
    print('world')

    config.DEBUG = True

    # Should stop execution here
    breakpoint()
    print('!!!')
