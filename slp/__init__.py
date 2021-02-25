import io
import logging

from typing import Optional, Any
from loguru import logger
from tqdm import tqdm


def configure_logging(logfile_prefix: Optional[str] = None) -> None:
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

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logger.info("Intercepting standard logging logs in loguru")

    # Make loguru play well with tqdm
    logger.remove()

    def tqdm_write(msg: str) -> Any:
        return tqdm.write(msg, end="")

    logger.add(tqdm_write, colorize=True)

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

    if logfile_prefix is not None:
        log_to_file(logfile_prefix)
