from tqdm import tqdm
from toolz.functoolz import compose
from torch.utils.data import Dataset


class LMDataset(Dataset):
    """Wraps a wikitext dataset from pytorch
    NLP which is provided as a list of tokens
    """
    def __init__(self, tokens, max_len=256):
        self.max_len = max_len

        self.data = [self._split_samples(tokens, idx)
                     for idx in tqdm(range(len(tokens) - 1), total=len(tokens) - 1)]
        self.transforms = []

    def _split_samples(self, tokens, idx):
        _len = min(self.max_len, len(tokens) - 1 - idx)
        inputs = tokens[idx:idx + _len]
        targets = tokens[idx + 1:idx + 1 + _len]
        return inputs, targets

    def map(self, fn, lazy=True):
        if lazy:
            self.transforms.append(fn)
        else:
            self.data = [(fn(d[0]), fn(d[1])) for d in self.data]
        return self

    def apply_transforms(self):
        fn = compose(*self.transforms[::-1])
        self.data = [(fn(d[0]), fn(d[1])) for d in tqdm(self.data, total=len(self.data))]
        self.transforms = []
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        for t in self.transforms:
            datum = t(datum)
        return datum
