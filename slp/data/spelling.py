import torch
from torch.utils.data import Dataset

from slp.data.transforms import ToTensor


class SpellCorrectorDataset(Dataset):
    def __init__(self, fname, tokenizer=None):
        self.data = []
        with open(fname, "r", errors="ignore") as fd:
            for l in fd:
                try:
                    dat = l.strip().split("\t")

                    if len(dat) == 2 and len(dat[1]) > 2 and len(dat[1]) < 256:
                        self.data.append(dat)
                except:
                    print(l)
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
