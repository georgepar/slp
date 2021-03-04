import io
import logging

from typing import Optional, Any
from loguru import logger
from tqdm import tqdm

from slp.util.system import date_fname


def log_to_file(fname_prefix: Optional[str]) -> str:
    """log_to_file Configure loguru to log to a logfile

    Args:
        logfile_prefix (Optional[str]): Optional prefix to file where logs will be written.

    Returns:
        str: The logfile where logs are written
    """
    logfile = f"{fname_prefix}.{date_fname()}.log"
    logger.add(
        logfile,
        colorize=False,
        level="DEBUG",
        enqueue=True,
    )
    return logfile


def configure_logging(logfile_prefix: Optional[str] = None) -> str:
    """configure_logging Configure loguru to intercept logging module logs, tqdm.writes and write to a logfile

    We use logure for stdout/stderr logging in this project.
    This function configures loguru to intercept logs from other modules that use the default python logging module.
    It also configures loguru so that it plays well with writes in the tqdm progress bars
    If a logfile_prefix is provided, loguru will also write all logs into a logfile with a unique name constructed using
    logfile_prefix and datetime.now()

    Args:
        logfile_prefix (Optional[str]): Optional prefix to file where logs will be written.

    Returns:
        str: The logfile where logs are written

    Examples:
        >>> configure_logging("logs/my-cool-experiment)
        logs/my-cool-experiment.20210228-211832.log
    """

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            """Intercept standard logging logs in loguru. Should test this for distributed pytorch lightning"""
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
        """tqdm write wrapper for loguru"""
        return tqdm.write(msg, end="")

    logger.add(tqdm_write, colorize=True)

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

    if logfile_prefix is not None:
        logfile = log_to_file(logfile_prefix)
        logger.info(f"Log file will be saved in {logfile}")

    return logfile
