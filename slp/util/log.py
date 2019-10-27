import logging
import logging.config
import logging.handlers

import slp.config as config

# Patch to avoid annoying ignite logs
logging.getLogger('ignite').setLevel(logging.WARNING)

logging.config.dictConfig(config.DEFAULT_LOGGING)

LOGGER = logging.getLogger('default')

debug = LOGGER.debug
info = LOGGER.info
warning = LOGGER.warning
warn = LOGGER.warn
error = LOGGER.error
critical = LOGGER.critical


if __name__ == '__main__':
    name = 'test'
    log.info('hello')
    log.debug('hello')
