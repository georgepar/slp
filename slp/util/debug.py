import sys

import slp.config as config
import slp.util.log as log

LOGGER = log.getLogger('slp')


if config.REMOTE_DEBUGGING:
    # import web_pdb as pdb
    # Use socket-based rpdb because web_pdb cannot handle large structures
    try:
        import rpdb as pdb
    except ImportError:
        LOGGER.warning('rpdb is not installed.'
                       'Remote debugging not available')
else:
    import pdb


def set_trace():
    if config.DEBUG:
        if config.REMOTE_DEBUGGING:
            pdb.set_trace(addr=config.DEBUG_ADDR, port=config.DEBUG_PORT)
        else:
            pdb.set_trace()


sys.breakpointhook = set_trace


if __name__ == '__main__':
    print('hello')

    config.DEBUG = False

    # Should not stop execution here
    breakpoint()

    print('world')

    config.DEBUG = True

    # Should stop execution here
    breakpoint()

    print('!!!')
