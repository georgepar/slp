from loguru import logger
from tqdm import tqdm
import io

# Make loguru play well with tqdm
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

