from mmsdk import mmdatasdk
from slp.mm.read_dataset import ReadDlData
import numpy as np
import os
import sys


def add_labels(cmumosi_higlevel, cmumosi_path, aligned_path):
    cmumosi_highlevel.add_computational_sequences(
        mmdatasdk.cmu_mosi.labels, cmumosi_path
    )
    cmumosi_highlevel.align("Opinion Segment Labels")
    deploy_files = {x: x for x in cmumosi_highlevel.computational_sequences.keys()}
    cmumosi_highlevel.deploy(aligned_path, deploy_files)
    aligned_cmumosi_highlevel = mmdatasdk.mmdataset(aligned_path)
    return cmumosi_higlevel


def myavg2(intervals, features):
    return np.average(features, axis=0)


def myavg(intervals, features):
    l = 5032  # 68 * 74
    features = features.reshape(1, -1)
    real_len = features.size
    pad_l = l - real_len
    if pad_l > 0:
        real_len = np.array(real_len).reshape(1, -1)
        zeros = np.zeros((1, pad_l))
        features = np.concatenate((features, zeros, real_len), axis=1)
    else:
        real_len = np.array(l).reshape(1, -1)
        features = np.concatenate((features[:, :l], real_len), axis=1)
    return features


def directory_check(dirName):
    exists = True
    if os.path.exists(dirName) and os.path.isdir(dirName):
        if not os.listdir(dirName):
            print("MOSI Directory is empty")
            exists = False
        else:
            print("MOSI Directory is not empty")
    else:
        print("MOSI Directory doesn't exist")
        exists = False
    return exists


BASE_PATH = sys.argv[1]
cmumosi_path = os.path.join(BASE_PATH, "cmumosi")
aligned_path = os.path.join(BASE_PATH, "aligned")
print("preaparing CMU MOSI download in {} and {}".format(cmumosi_path, aligned_path))

if not directory_check(cmumosi_path):
    ## execute the following command only once
    print(".....mosi is being downloaded")
    cmumosi_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, cmumosi_path)
else:
    print("mosi is already downloaded")
    cmumosi_highlevel = ReadDlData(mmdatasdk.cmu_mosi.highlevel, cmumosi_path)


cmumosi_highlevel.align("glove_vectors", collapse_functions=[myavg])
cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels, cmumosi_path)
size_list = [9216, 74, 47, 300, 1585]

cmumosi_highlevel.align("Opinion Segment Labels")
deploy_files = {x: x for x in cmumosi_highlevel.computational_sequences.keys()}
cmumosi_highlevel.deploy(aligned_path, deploy_files)
aligned_cmumosi_highlevel = mmdatasdk.mmdataset(aligned_path)


# cmumosi_highlevel = add_labels(cmumosi_highlev)
