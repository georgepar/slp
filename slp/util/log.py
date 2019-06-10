import logging
import logging.config
import logging.handlers

import slp.config as config

# Patch to avoid annoying ignite logs
logging.getLogger("ignite").setLevel(logging.WARNING)


def getLogger(name):
    logging.config.dictConfig(config.DEFAULT_LOGGING)
    log = logging.getLogger(name)
    return log


if __name__ == '__main__':
    name = 'test'
    log = getLogger(name)
    log.info('hello')
    log.debug('hello')
