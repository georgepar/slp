import sys

import slp.util.log as log

LOGGER = log.getLogger('slp')

try:
    import web_pdb as pdb
except ImportError:
    LOGGER.warning('web_pdb is not installed.'
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
    breakpoint()
    print('world')

    config.DEBUG = True

    # Should stop execution here
    breakpoint()
    print('!!!')
