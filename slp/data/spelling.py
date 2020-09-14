import os

import torch
from torch.utils.data import Dataset

from slp.data.transforms import ToTensor


class SpellCorrectorDataset(Dataset):
    def __init__(self, fname, tokenizer=None, max_length=256):
        self.data = []

        with open(fname, "r", errors="ignore") as fd:
            for ln in fd:
                try:
                    dat = ln.strip().split("\t")

                    if (
                        len(dat) == 2
                        and len(dat[1]) > 2
                        and len(dat[1]) < max_length
                        and len(dat[0]) < max_length
                    ):
                        self.data.append(dat)
                except:
                    pass
            # self.data = [line.strip().split("\t") for line in fd]
        print("Read {} lines".format(len(self.data)))
        self.tokenizer = tokenizer
        self.tt = ToTensor(device="cpu", dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = self.tt(self.tokenizer(src))
        tgt = self.tt(self.tokenizer(tgt))

        return src, tgt


class SpellCorrectorPredictionDataset(Dataset):
    def __init__(self, fname, tokenizer=None, max_length=256):
        self.data = []

        with open(fname, "r", errors="ignore") as fd:
            for ln in fd:
                try:
                    dat = ln.strip()
                    self.data.append(dat)
                except:
                    pass
        self.tokenizer = tokenizer
        self.tt = ToTensor(device="cpu", dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]
        src = self.tt(self.tokenizer(src))
        tgt = self.tt(self.tokenizer(""))

        return src, tgt
