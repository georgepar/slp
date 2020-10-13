import mmsdk
import os
import re
import numpy as np
from mmsdk import mmdatasdk as md
from subprocess import check_call, CalledProcessError
import sys

DATA_PATH = sys.argv[1]

# create folders for storing the data
if not os.path.exists(DATA_PATH):
    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

# download highlevel features, low-level (raw) data and labels for the dataset MOSI
# if the files are already present, instead of downloading it you just load it yourself.
# here we use CMU_MOSI dataset as example.

DATASET = md.cmu_mosi

try:
    md.mmdataset(DATASET.highlevel, DATA_PATH)
except RuntimeError:
    print("High-level features have been downloaded previously.")

try:
    md.mmdataset(DATASET.raw, DATA_PATH)
except RuntimeError:
    print("Raw data have been downloaded previously.")

try:
    md.mmdataset(DATASET.labels, DATA_PATH)
except RuntimeError:
    print("Labels have been downloaded previously.")

