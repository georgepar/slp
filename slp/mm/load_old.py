import errno
import glob
import os
import numpy as np

from mmsdk import mmdatasdk
from slp.util import log
from slp.util import h52np


def load_csds(path, modality_files):
    if not os.path.isdir(path):
        log.error(
            "{} does not exist. You must download data first"
            .format(path)
        )
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    all_csds = map(os.path.basename, glob.glob(os.path.join(path, "*.csd")))
    csds = [f for f in all_csds if f in modality_files]

    if len(csds) == 0:
        log.error("No csd files in {}".format(path))
        raise ValueError("No csds found")

    datadict = {csd: os.path.join(path, csd) for csd in csds}

    data = mmdatasdk.mmdataset(datadict)
    log.info("Loaded computational sequences for MOSI")
    log.info(data.computational_sequences.keys())
    return data


def load_mosi(base_path, aligned=True, binary=True):
    folder = 'cmumosi' if not aligned else 'aligned'
    path = os.path.join(base_path, folder)

    MODALITIES = [
        'glove_vectors.csd',
        'COVAREP.csd',
        'Opinion Segment Labels.csd'
    ]

    data = load_csds(path, MODALITIES)
    glove, covarep, opinions = [
        data.computational_sequences[csd].data for csd in MODALITIES
    ]

    def mmfeats(k):
        return (
            h52np(glove[k]['features']),
            h52np(covarep[k]['features']),
            h52np(opinions[k]['features'])
        )

    allfeats = [mmfeats(k) for k in glove.keys()]
    if binary:
        # kick-out unsigned-neutral samples
        allfeats = [(t, a, o) for t, a, o in allfeats if np.sign(o) != 0]
    txt, audio, opns = zip(*allfeats)

    dataset = {
        'audio': audio,
        'text': txt,
        'opinions': opns
    }
    return dataset


if __name__ == '__main__':
    import sys
    data = load_mosi(sys.argv[1], binary=False)
    import ipdb; ipdb.set_trace()
