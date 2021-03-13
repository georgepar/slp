import numpy as np
from torch.utils.data import Dataset


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
