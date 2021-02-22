import io
import logging

from loguru import logger
from tqdm import tqdm


def configure_logger(logfile_prefix):
    from slp.util.system import log_to_file


    # Intercept standard logging logs in loguru. Should test this for distributed pytorch lightning
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logger.debug("Intercepting standard logging logs in loguru")
    logger.debug("This is a side-effect")

    # Make loguru play well with tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    log_to_file(logfile_prefix)
