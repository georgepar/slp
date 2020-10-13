import os
from mmsdk import mmdatasdk
from os import listdir
from os.path import isfile, join


def InverseMap(dict):
    inv_map = {v: k for k, v in dict.items()}
    return inv_map


def SearchDictKeys(dict, name_list):
    short_names = {}
    for csd_name in name_list:
        for k in dict:
            if csd_name in k:
                short_names[csd_name] = dict[k]
                continue
    return short_names


def ReadDlData(name_dict, path):
    """Function that read already downloaded files
    from specified given path which is given as
    input argument. Returns a mmdatasdk class with
    all available multimodal data features"""

    url_dictionary = InverseMap(name_dict)
    dataset_dictionary = {}

    if os.path.isdir(path) is False:
        print("Folder does not exist ...")
        exit(-1)

    csdfiles = [f for f in listdir(path) if isfile(join(path, f)) and f[-4:] == ".csd"]
    if len(csdfiles) == 0:
        print("No csd files in the given folder")
        exit(-2)

    highlevel_names = SearchDictKeys(url_dictionary, csdfiles)

    print("%d csd files found" % len(csdfiles))
    for csdfile in csdfiles:
        dataset_dictionary[highlevel_names[csdfile]] = os.path.join(path, csdfile)
    dataset = mmdatasdk.mmdataset(dataset_dictionary)

    print("List of the computational sequences")
    print(dataset.computational_sequences.keys())

    return dataset
