from tqdm import tqdm
from toolz.functoolz import compose
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class CorpusLMDataset(Dataset):
    """Wraps a wikitext dataset from pytorch
    NLP which is provided as a list of tokens
    """

    def __init__(self, corpus):
        self.source = corpus[:-1]
        self.target = corpus[1:]
        self.transforms = []

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return int(len(self.source))

    def __getitem__(self, idx):
        src, tgt = self.source[idx], self.target[idx]
        for t in self.transforms:
            src = t(src)
            tgt = t(tgt)
        return src, tgt


class CorpusDataset(Dataset):
    def __init__(self, corpus, labels):
        self.corpus = corpus
        self.labels = labels
        assert len(self.labels) == len(self.corpus), "Incompatible labels and corpus"
        self.transforms = []
        self.label_encoder = LabelEncoder().fit(self.labels)

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text, target = self.corpus[idx], self.labels[idx]
        target = self.label_encoder.transform([target])[0]
        for t in self.transforms:
            text = t(text)
        return text, target
