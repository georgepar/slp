import copy
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import torch
from toolz import compose_left, pipe
from torch.utils.data import Dataset
from tqdm import tqdm

from slp.data.transforms import ToTensor


class MMDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        modalities: Union[List[str], Set[str]] = {"text", "audio", "visual"},
    ):
        self.data = data
        self.modalities = set(list(modalities) + ["label"])

        self.transforms: Dict[str, List[Callable]] = {m: [] for m in self.modalities}
        self.transforms["label"] = []

    def map(self, fn: Callable, modality: str, lazy: bool = True):
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
            fn = compose_left(*self.transforms[m])
            # In place transformation to save some mem.

            for i in tqdm(
                range(len(self.data)),
                desc=f"Applying transforms for {m}",
                total=len(self.data),
            ):
                self.data[i][m] = fn(self.data[i][m])
        self.transforms = {m: [] for m in self.modalities}

        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dat = {m: self.data[idx][m] for m in self.modalities}
        for m in self.modalities:
            if len(self.transforms[m]) == 0:
                continue
            dat[m] = pipe(copy.deepcopy(dat[m]), *self.transforms[m])

        return dat


def binarize(x):
    return 0.5 * (1.0 + np.sign(x)).astype(int)


class MOSI(MMDataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        modalities: Union[List[str], Set[str]] = {"text", "audio", "visual"},
        text_is_tokens: bool = False,
        binary: bool = False,
    ):
        super(MOSI, self).__init__(data, modalities)

        def label_selector(l):
            return l.item()

        self.map(label_selector, "label", lazy=True)

        if binary:
            self.map(binarize, "label", lazy=True)

        for m in self.modalities:
            if m == "text" and text_is_tokens:
                self.map(ToTensor(dtype=torch.long), m, lazy=True)
            else:
                self.map(ToTensor(dtype=torch.float), m, lazy=True)


class MOSEI(MMDataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        modalities: Union[List[str], Set[str]] = {"text", "audio", "visual"},
        text_is_tokens: bool = False,
        label_selector: Optional[Callable] = None,
    ):
        super(MOSEI, self).__init__(data, modalities)

        def default_label_selector(l):
            return l[0][0]

        if label_selector is None:
            label_selector = default_label_selector

        self.map(label_selector, "label", lazy=True)

        for m in self.modalities:
            if m == "text" and text_is_tokens:
                self.map(ToTensor(dtype=torch.long), m, lazy=True)
            else:
                self.map(ToTensor(dtype=torch.float), m, lazy=True)
