import copy
import gc

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from cytoolz.functoolz import compose
except ImportError:
    from toolz.functoolz import compose


class MOSI(Dataset):
    def __init__(
        self, data, binary=False, modalities={"text", "audio"}, transforms=None
    ):
        self.data = data

        if binary:
            self.data = self.binarize(self.data)
        self.modalities = modalities
        self.transforms = transforms

        if self.transforms is None:
            self.transforms = {m: [] for m in self.modalities}

    def binarize(self, data):
        for i in range(len(data)):
            data[i]["label"] = 0.5 * (1 + np.sign(data[i]["label"])).astype(int)

        return data

    def map(self, fn, modality, lazy=True):
        if modality not in self.modalities:
            return self
        self.transforms[modality].append(fn)

        if not lazy:
            self.apply_transforms()

        return self

    def apply_transforms(self):
        for m in self.modalities:
            if len(self.transforms[m]) == 0:
                continue
            fn = compose(*self.transforms[m][::-1])
            # In place transformation to save some mem.

            for i in tqdm(range(len(self.data)), total=len(self.data)):
                self.data[i][m] = fn(self.data[i][m])
        self.transforms = {m: [] for m in self.modalities}

        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = self.data[idx]

        return dat


class MOSIMFN(Dataset):
    def __init__(
        self,
        X,
        y,
        binary=True,
        unpad=True,
        lmf=True,
        modalities={"text", "audio"},
        transforms=None,
    ):
        if unpad:
            self.X = self.unpad(X)
        else:
            self.X = X
        self.y = y
        self.get_mods = self.get_mods_lmf if lmf else self.get_mods_efthymis
        self.data = [self.get_mods(i) for i in range(len(y))]
        self.binary = binary

        if binary:
            self.data = self.binarize(self.data)
            self.y = [(0.5 * (1 + np.sign(l))).astype(int) for l in self.y]
        self.modalities = modalities
        self.transforms = transforms

        if self.transforms is None:
            self.transforms = {m: [] for m in self.modalities}

    def get_mods_lmf(self, idx):
        return {
            "text": self.X[idx][:, :300],
            "audio": self.X[idx][:, 300:305],
            "visual": self.X[idx][:, 305:],
            "length": self.X[idx].shape[0],
            "label": self.y[idx],
        }

    def get_mods_efthymis(self, idx):
        return {
            "text": self.X[idx][:, :300],
            "audio": self.X[idx][:, 300:],
            "length": self.X[idx].shape[0],
            "label": self.y[idx],
        }

    def binarize(self, data):
        for i in range(len(data)):
            data[i]["label"] = 0.5 * (1 + np.sign(data[i]["label"])).astype(int)

        return data

    def unpad(self, X):
        data = []

        for i, x in enumerate(X):
            idx = 0

            for seg in x:
                if sum(seg) == 0:
                    idx += 1
                else:
                    break
            data.append(X[i, idx:, :])

        return data

    def map(self, fn, modality, lazy=True):
        if modality not in self.modalities:
            return self
        self.transforms[modality].append(fn)

        if not lazy:
            self.apply_transforms()

        return self

    def apply_transforms(self):
        for m in self.modalities:
            if len(self.transforms[m]) == 0:
                continue
            fn = compose(*self.transforms[m][::-1])
            # In place transformation to save some mem.

            for i in tqdm(range(len(self.data)), total=len(self.data)):
                self.data[i][m] = fn(self.data[i][m])
        self.transforms = {m: [] for m in self.modalities}

        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = self.data[idx]

        if self.binary:
            dat["label"] = int(dat["label"])

        return dat


class MOSEI(Dataset):
    def __init__(
        self,
        data,
        select_label=None,
        unpad=True,
        modalities={"text", "audio"},
        transforms=None,
    ):
        data1 = {k: [] for k in data[0].keys()}
        self.data = []

        for dat in data:
            for k, v in dat.items():
                data1[k].append(v)
        data = data1
        self.select_label = select_label
        self.labels = data["label"]

        if unpad:
            self.unpad_idx = self.unpad_indices(data["text"])

        for i in range(len(self.labels)):
            dat = {}

            for k in modalities:
                if unpad:
                    dat[k] = data[k][i][self.unpad_idx[i] :, ...]
                else:
                    dat[k] = data[k][i]
            self.data.append(dat)
        gc.collect()
        self.modalities = modalities
        self.transforms = transforms

        if self.transforms is None:
            self.transforms = {m: [] for m in self.modalities}

    def map(self, fn, modality, lazy=True):
        if modality not in self.modalities:
            return self
        self.transforms[modality].append(fn)

        if not lazy:
            self.apply_transforms()

        return self

    def unpad_indices(self, dat):
        indices = []

        for x in dat:
            idx = 0

            for seg in x:
                if sum(seg) == 0:
                    idx += 1
                else:
                    break
            indices.append(idx)

        return indices

    def apply_transforms(self):
        for m in self.modalities:
            if len(self.transforms[m]) == 0:
                continue
            fn = compose(*self.transforms[m][::-1])
            # In place transformation to save some mem.

            for i in tqdm(range(len(self.data)), total=len(self.data)):
                self.data[i][m] = fn(self.data[i][m])
        self.transforms = {m: [] for m in self.modalities}

        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = self.data[idx]
        dat["label"] = self.labels[idx]

        if self.select_label is not None:
            dat["label"] = dat["label"][self.select_label]

        return dat


class MOSEI_MULT(Dataset):
    def __init__(self, data, modalities={"text", "audio", "visual"}, transforms=None):
        visual = data["vision"].astype(np.float32)
        audio = data["audio"].astype(np.float32)
        audio[audio == -np.inf] = 0
        audio[audio == np.inf] = 0
        text = data["text"].astype(np.float32)
        self.labels = data["labels"].astype(np.float32).squeeze(-1)
        self.data = []

        for v, a, t in zip(visual, audio, text):
            dat = {
                "visual": v,
                "audio": a,
                "text": t,
            }
            self.data.append(dat)

        self.modalities = modalities
        self.transforms = transforms

        if self.transforms is None:
            self.transforms = {m: [] for m in self.modalities}

    def map(self, fn, modality, lazy=True):
        if modality not in self.modalities:
            return self
        self.transforms[modality].append(fn)

        if not lazy:
            self.apply_transforms()

        return self

    def apply_transforms(self):
        for m in self.modalities:
            if len(self.transforms[m]) == 0:
                continue
            fn = compose(*self.transforms[m][::-1])
            # In place transformation to save some mem.

            for i in tqdm(range(len(self.data)), total=len(self.data)):
                self.data[i][m] = fn(self.data[i][m])
        self.transforms = {m: [] for m in self.modalities}

        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dat = self.data[idx]
        dat["label"] = self.labels[idx]

        return dat
