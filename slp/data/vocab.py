import itertools
from collections import Counter
from enum import Enum

from slp.config import SPECIAL_TOKENS


def create_vocab(corpus, vocab_size=5000, extra_tokens=SPECIAL_TOKENS):
    if isinstance(corpus[0], list):
        corpus = itertools.chain.from_iterable(corpus)
    freq = Counter(corpus)
    if isinstance(extra_tokens, Enum):
        extra_tokens = extra_tokens.to_list()
    take = min(vocab_size - len(extra_tokens), len(freq))
    common_words = list(map(lambda x: x[0], freq.most_common(take)))
    words = extra_tokens + common_words
    vocab = dict(zip(words, itertools.count()))
    return vocab
