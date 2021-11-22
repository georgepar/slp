import io
import os
import pickle
from typing import List

import lmdb
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class CorpusLMDataset(Dataset):
    def __init__(self, corpus):
        """Wraps a tokenized dataset which is provided as a list of tokens

        Targets = source shifted one token to the left (next token prediction)

        Args:
            corpus (List[str] or WordCorpus): List of tokens
        """
        self.source = corpus[:-1]
        self.target = corpus[1:]
        self.transforms = []

    def map(self, t):
        """Append a transform to self.transforms, in order to be applied to the data

        Args:
            t (Callable[[str], Any]): Transform of input token

        Returns:
            CorpusLMDataset: self
        """
        self.transforms.append(t)

        return self

    def __len__(self):
        """Length of corpus

        Returns:
            int: Corpus Length
        """

        return int(len(self.source))

    def __getitem__(self, idx):
        """Get a source and target token from the corpus

        Args:
            idx (int): Token position

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: source=coprus[idx], target=corpus[idx+1]
        """
        src, tgt = self.source[idx], self.target[idx]

        for t in self.transforms:
            src = t(src)
            tgt = t(tgt)

        return src, tgt


class CorpusDataset(Dataset):
    def __init__(self, corpus, labels):
        """Labeled corpus dataset

        Args:
            corpus (WordCorpus, HfCorpus etc..): Input corpus
            labels (List[Any]): Labels for examples
        """
        self.corpus = corpus
        self.labels = labels
        assert len(self.labels) == len(self.corpus), "Incompatible labels and corpus"
        self.transforms = []
        self.label_encoder = None

        if isinstance(self.labels[0], str):
            self.label_encoder = LabelEncoder().fit(self.labels)

    def map(self, t):
        """Append a transform to self.transforms, in order to be applied to the data

        Args:
            t (Callable[[str], Any]): Transform of input token

        Returns:
            CorpusDataset: self
        """
        self.transforms.append(t)

        return self

    def __len__(self):
        """Length of corpus

        Returns:
            int: Corpus Length
        """

        return len(self.corpus)

    def __getitem__(self, idx):
        """Get a source and target token from the corpus

        Args:
            idx (int): Token position

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (processed sentence, label)
        """
        text, target = self.corpus[idx], self.labels[idx]

        if self.label_encoder is not None:
            target = self.label_encoder.transform([target])[0]

        for t in self.transforms:
            text = t(text)

        return text, target


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=os.path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = self.loads_data(txn.get(b"__len__"))
            self.keys = self.loads_data(txn.get(b"__keys__"))

        self.transform = transform
        self.target_transform = target_transform

    def loads_data(self, buf):
        return pickle.loads(buf)

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = self.loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target

        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"
