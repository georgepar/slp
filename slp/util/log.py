import logging
import logging.config
import logging.handlers
import slp.config.log as log_config


def getLogger(name):
    logging.config.dictConfig(log_config.DEFAULT_LOGGING)
    log = logging.getLogger(name)
    return log


if __name__ == '__main__':
    name = 'test'
    log = getLogger(name)
    log.info('hello')
    log.debug('hello')
