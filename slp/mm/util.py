import mmsdk
from mmsdk import mmdatasdk as md

from slp.util.system import safe_mkdirs
from slp.util import log


def download_mmdata(base_path, dataset):
    safe_mkdirs(base_path)

    try:
        md.mmdataset(dataset.highlevel, base_path)
    except RuntimeError:
        log.info("High-level features have been downloaded previously.")

    try:
        md.mmdataset(dataset.raw, base_path)
    except RuntimeError:
        log.info("Raw data have been downloaded previously.")

    try:
        md.mmdataset(dataset.labels, base_path)
    except RuntimeError:
        log.info("Labels have been downloaded previously.")
