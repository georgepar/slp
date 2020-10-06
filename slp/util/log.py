import logging
import logging.config
import logging.handlers
import os

import slp.config.log as log_config

# Patch to avoid annoying ignite logs
logging.getLogger("ignite").setLevel(logging.WARNING)


def mklogger(filename=None):
    if filename is not None:
        logfile = os.path.join(log_config.LOG_PATH, filename)
        log_config.DEFAULT_LOGGING["handlers"]["logfile"]["filename"] = logfile
    print(f"Logfile: {logfile}")
    logging.config.dictConfig(log_config.DEFAULT_LOGGING)
    LOGGER = logging.getLogger("default")

    return LOGGER
