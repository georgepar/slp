import itertools
from collections import Counter


def create_vocab(corpus, vocab_size=5000, extra_tokens=None):
    if isinstance(corpus[0], list):
        corpus = itertools.chain.from_iterable(corpus)
    freq = Counter(corpus)
    if extra_tokens is None:
        extra_tokens = []
    take = min(vocab_size, len(freq))
    common_words = list(map(lambda x: x[0], freq.most_common(take)))
    common_words = list(set(common_words) - set(extra_tokens))
    words = extra_tokens + common_words
    if len(words) > vocab_size:
        words = words[:vocab_size]
    vocab = dict(zip(words, itertools.count()))
    return vocab
