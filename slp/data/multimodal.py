import copy
import random
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import torch
from slp.data.transforms import ToTensor
from toolz import compose_left, pipe
from torch.utils.data import Dataset
from tqdm import tqdm


class MissingModalities(object):
    def __init__(
        self,
        modalities={"text", "audio", "visual"},
        modality_to_miss=None,
        frame_drop_percentage=1.0,
        repeat_frame=False,
    ):
        self.modality_to_miss = modality_to_miss
        self.modalities = list(modalities)
        self.frame_drop_percentage = frame_drop_percentage
        self.repeat_frame = repeat_frame

    def __call__(self, x):
        m2d = (
            self.modality_to_miss

            if self.modality_to_miss is not None
            else random.choice(self.modalities)
        )
        seqlen = x[m2d].shape[0]

        for i in range(seqlen):
            if random.random() < self.frame_drop_percentage:
                if self.repeat_frame and i > 0:
                    x[m2d][i, :] = x[m2d][i - 1, :]
                else:
                    x[m2d][i, :] = x[m2d][i, :] * 0.0

        return x


class MMDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        modalities: Union[List[str], Set[str]] = {"text", "audio", "visual"},
        missing_modalities=False,
        modality_to_miss=None,
        frame_drop_percentage=1.0,
        repeat_frame=False,
    ):
        self.data = data
        self.modalities = set(list(modalities) + ["label"])

        self.transforms: Dict[str, List[Callable]] = {m: [] for m in self.modalities}
        self.transforms["label"] = []

        self.missing_modalities = None

        if missing_modalities:
            self.missing_modalities = MissingModalities(
                modalities=modalities,
                modality_to_miss=modality_to_miss,
                frame_drop_percentage=frame_drop_percentage,
                repeat_frame=repeat_frame,
            )

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

        if self.missing_modalities is not None:
            dat = self.missing_modalities(dat)

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
        missing_modalities=False,
        modality_to_miss=None,
        frame_drop_percentage=1.0,
        repeat_frame=False,
    ):
        super(MOSI, self).__init__(
            data,
            modalities,
            missing_modalities=missing_modalities,
            modality_to_miss=modality_to_miss,
            frame_drop_percentage=frame_drop_percentage,
            repeat_frame=repeat_frame,
        )

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
        missing_modalities=False,
        modality_to_miss=None,
        frame_drop_percentage=1.0,
        repeat_frame=False,
    ):
        super(MOSEI, self).__init__(
            data,
            modalities,
            missing_modalities=missing_modalities,
            modality_to_miss=modality_to_miss,
            frame_drop_percentage=frame_drop_percentage,
            repeat_frame=repeat_frame,
        )

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
