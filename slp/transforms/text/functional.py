import numpy as np
import re
from sklearn.utils import check_array


def strip_punctuation(s):
    return re.sub(r'[^a-zA-Z\s]', ' ', s)


def preprocess(s):
    # flake8: noqa W605
    return re.sub('\s+', ' ', strip_punctuation(s).lower())


def split_tokenizer(x):
    return x.split(' ')


def untokenize(x):
    return " ".join(x)


def gaussian_noise(X, mu=0.0, std=0.1):
    X = check_array(X)
    return X + np.random.normal(mu, std, X.shape)


def aggregate_vecs(vectors, aggregation='mean'):
    if isinstance(aggregation, str):
        aggregation = [aggregation]
    feats = []
    for method in aggregation:
        if method == "sum":
            feats.append(np.sum(vectors, axis=0))
        if method == "mean":
            feats.append(np.mean(vectors, axis=0))
        if method == "min":
            feats.append(np.amin(vectors, axis=0))
        if method == "max":
            feats.append(np.amax(vectors, axis=0))
    return np.hstack(feats)
