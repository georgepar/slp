import itertools
from collections import Counter

import errno
import os

import numpy as np
from loguru import logger
from enum import Enum

from tqdm import tqdm
from typing import cast, Any, Dict, Optional, List

from slp.data.transforms import SpacyTokenizer, WordpieceTokenizer, ToTokenIds
from slp.util import system
from slp.util import types
from slp.config import SPECIAL_TOKENS


def create_vocab(corpus, vocab_size=-1, extra_tokens: Optional[List] = None):
    if isinstance(corpus[0], list):
        corpus = itertools.chain.from_iterable(corpus)
    freq = Counter(corpus)
    if extra_tokens is None:
        extra_tokens = []
    else:
        extra_tokens = extra_tokens.to_list()
    if vocab_size < 0:
        vocab_size = len(freq)
    take = min(vocab_size, len(freq))
    logger.info(f"Keeping {vocab_size} most common tokens out of {len(freq)}")
    common_words = list(map(lambda x: x[0], freq.most_common(take)))
    common_words = list(set(common_words) - set(extra_tokens))
    words = extra_tokens + common_words
    if len(words) > vocab_size:
        words = words[: vocab_size + len(extra_tokens)]

    def token_freq(t):
        return 0 if t in extra_tokens else freq[t]

    vocab = dict(zip(words, map(token_freq, words)))
    logger.info(f"Vocabulary created with {len(vocab)} tokens.")
    logger.info(f"The 10 most common tokens are:\n{freq.most_common(10)}")

    return vocab


class EmbeddingsLoader(object):
    def __init__(
        self,
        embeddings_file: str,
        dim: int,
        vocab: Optional[Dict[str, int]] = None,
        extra_tokens: Any = SPECIAL_TOKENS,
    ) -> None:
        self.embeddings_file = embeddings_file
        self.vocab = vocab
        self.cache_ = self._get_cache_name()
        self.dim_ = dim
        self.extra_tokens = extra_tokens

    def in_accepted_vocab(self, word):
        if self.vocab is None:
            return True
        else:
            return word in self.vocab

    def _get_cache_name(self) -> str:
        head, tail = os.path.split(self.embeddings_file)
        filename, ext = os.path.splitext(tail)
        cache_name = os.path.join(head, f"{filename}.{len(self.vocab)}.p")
        logger.info(f"Cache: {cache_name}")
        return cache_name

    def _dump_cache(self, data: types.Embeddings) -> None:
        system.pickle_dump(data, self.cache_)

    def _load_cache(self) -> types.Embeddings:
        return cast(types.Embeddings, system.pickle_load(self.cache_))

    def augment_embeddings(
        self,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
        embeddings: np.ndarray,
        token: str,
        emb: Optional[np.ndarray] = None,
    ) -> types.Embeddings:
        word2idx[token] = len(embeddings)
        idx2word[len(embeddings)] = token
        if emb is None:
            emb = np.random.uniform(low=-0.05, high=0.05, size=self.dim_)
        embeddings.append(emb)
        return word2idx, idx2word, embeddings

    @system.timethis
    def load(self) -> types.Embeddings:
        """
        Read the word vectors from a text file
        Returns:
            word2idx (dict): dictionary of words to ids
            idx2word (dict): dictionary of ids to words
            embeddings (numpy.ndarray): the word embeddings matrix
        """
        # in order to avoid this time consuming operation, cache the results
        try:
            cache = self._load_cache()
            logger.info("Loaded word embeddings from cache.")
            return cache
        except OSError:
            logger.warning(f"Didn't find embeddings cache file {self.embeddings_file}")
            logger.warning("Loading embeddings from file.")

        # create the necessary dictionaries and the word embeddings matrix
        if not os.path.exists(self.embeddings_file):
            logger.critical(f"{self.embeddings_file} not found!")
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), self.embeddings_file)

        logger.info(f"Indexing file {self.embeddings_file} ...")

        # create the 2D array, which will be used for initializing
        # the Embedding layer of a NN.
        # We reserve the first row (idx=0), as the word embedding,
        # which will be used for zero padding (word with id = 0).
        word2idx, idx2word, embeddings = self.augment_embeddings(
            {}, {}, [], self.extra_tokens.PAD.value, emb=np.zeros(self.dim_)
        )

        for token in self.extra_tokens:
            logger.debug(f"Adding token {token.value} to embeddings matrix")
            if token == self.extra_tokens.PAD:
                continue
            word2idx, idx2word, embeddings = self.augment_embeddings(
                word2idx, idx2word, embeddings, token.value
            )

        # read file, line by line
        with open(self.embeddings_file, "r") as f:
            num_lines = sum(1 for line in f)

        with open(self.embeddings_file, "r") as f:
            index = len(embeddings)

            for line in tqdm(
                f, total=num_lines, desc="Loading word embeddings...", leave=False
            ):
                # skip the first row if it is a header
                if len(line.split()) < self.dim_:
                    continue

                values = line.rstrip().split(" ")
                word = values[0]

                if word in word2idx:
                    continue

                if not self.in_accepted_vocab(word):
                    continue

                vector = np.asarray(values[1:], dtype=np.float32)
                idx2word[index] = word
                word2idx[word] = index
                embeddings.append(vector)
                index += 1

        logger.info(f"Loaded {len(embeddings)} word vectors.")
        embeddings = np.array(embeddings, dtype="float32")

        # write the data to a cache file
        self._dump_cache((word2idx, idx2word, embeddings))
        return word2idx, idx2word, embeddings


class WordCorpus(object):
    def __init__(
        self,
        corpus,
        limit_vocab_size=30000,
        word2idx=None,
        idx2word=None,
        embeddings=None,
        embeddings_file=None,
        embeddings_dim=300,
        lower=True,
        special_tokens=SPECIAL_TOKENS,
        prepend_cls=False,
        prepend_bos=False,
        append_eos=False,
        lang="en_core_web_md",
        **kwargs,
    ):
        self.corpus_ = corpus
        self.tokenizer = SpacyTokenizer(
            lower=lower,
            prepend_cls=prepend_cls,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
            specials=special_tokens,
            lang=lang,
        )

        logger.info(f"Tokenizing corpus using spacy {lang}")

        self.tokenized_corpus_ = [
            self.tokenizer(s)
            for s in tqdm(self.corpus_, desc="Tokenizing corpus...", leave=False)
        ]

        self.vocab_ = create_vocab(
            self.tokenized_corpus_,
            vocab_size=limit_vocab_size if word2idx is None else -1,
            extra_tokens=special_tokens,
        )

        self.word2idx_, self.idx2word_, self.embeddings_ = None, None, None
        self.corpus_indices_ = self.tokenized_corpus_

        if embeddings_file is not None:
            logger.info(
                "Going to load {len(self.vocab_)} embeddings from {embeddings_file}"
            )
            loader = EmbeddingsLoader(
                embeddings_file,
                embeddings_dim,
                vocab=self.vocab_,
                extra_tokens=special_tokens,
            )
            self.word2idx_, self.idx2word_, self.embeddings_ = loader.load()

        if embeddings is not None:
            self.embeddings_ = embeddings

        if word2idx is not None:
            logger.info("Word2idx was already provided. Going to used it.")
            self.word2idx_ = word2idx

        if idx2word is not None:
            self.idx2word_ = idx2word

        if self.word2idx_ is not None:
            logger.info("Converting tokens to ids using word2idx.")
            self.to_token_ids = ToTokenIds(self.word2idx_, specials=SPECIAL_TOKENS)
            self.corpus_indices_ = [
                self.to_token_ids(s)
                for s in tqdm(
                    self.tokenized_corpus_,
                    desc="Converting tokens to token ids...",
                    leave=False,
                )
            ]

            logger.info("Filtering corpus vocabulary.")

            updated_vocab = {}
            for k, v in self.vocab_.items():
                if k in self.word2idx_:
                    updated_vocab[k] = v

            logger.info(
                "Out of {len(self.vocab_)} tokens {len(self.vocab_) - len(updated_vocab)} were not found in the pretrained embeddings."
            )

            self.vocab_ = updated_vocab

    @property
    def vocab_size(cls):
        return (
            cls.embeddings.shape[0] if cls.embeddings is not None else len(cls.vocab_)
        )

    @property
    def frequencies(cls):
        return cls.vocab_

    @property
    def vocab(cls):
        return set(cls.vocab_.keys())

    @property
    def embeddings(cls):
        return cls.embeddings_

    @property
    def word2idx(cls):
        return cls.word2idx_

    @property
    def idx2word(cls):
        return cls.idx2word_

    @property
    def tokenized(cls):
        return self.tokenized_corpus_

    @property
    def indices(cls):
        return self.corpus_indices_

    @property
    def raw(cls):
        return self.corpus_

    def __len__(self):
        return len(self.corpus_indices_)

    def __getitem__(self, idx):
        return self.corpus_indices_[idx]


class WordpieceCorpus(object):
    def __init__(
        self,
        corpus,
        lower=True,
        bert_model="bert-base-uncased",
        prepend_cls=False,
        prepend_bos=False,
        append_eos=False,
        special_tokens=SPECIAL_TOKENS,
        **kwargs,
    ):
        self.corpus_ = corpus

        logger.info(f"Tokenizing corpus using wordpiece tokenizer from {bert_model}")

        self.tokenizer = WordpieceTokenizer(
            lower=lower,
            bert_model=bert_model,
            prepend_bos=prepend_bos,
            prepend_cls=prepend_cls,
            append_eos=append_eos,
            specials=special_tokens,
        )

        self.tokenized_corpus_ = [
            self.tokenizer.tokenize(s)
            for s in tqdm(self.corpus_, desc="Tokenizing corpus...", leave=False)
        ]

        self.vocab_ = create_vocab(
            self.tokenized_corpus_,
            vocab_size=-1,
            extra_tokens=special_tokens,
        )
        self.corpus_indices_ = [
            self.tokenizer.to_ids(s)
            for s in tqdm(
                self.tokenized_corpus_,
                desc="Converting tokens to indices...",
                leave=False,
            )
        ]

    @property
    def vocab_size(cls):
        return cls.tokenizer.vocab_size

    @property
    def frequencies(cls):
        return cls.vocab_

    @property
    def embeddings(cls):
        return None

    @property
    def word2idx(cls):
        return None

    @property
    def idx2word(cls):
        return None

    @property
    def tokenized(cls):
        return self.tokenized_corpus_

    @property
    def indices(cls):
        return self.corpus_indices_

    @property
    def raw(cls):
        return self.corpus_

    def __len__(self):
        return len(self.corpus_indices_)

    def __getitem__(self, idx):
        return self.corpus_indices_[idx]


if __name__ == "__main__":
    corpus = [
        "the big",
        "brown fox",
        "jumps over",
        "the lazy dog",
        "supercalifragilisticexpialidocious",
    ]

    word_corpus = WordCorpus(
        corpus,
        embeddings_file="./cache/glove.6B.50d.txt",
        embeddings_dim=50,
        lower=True,
        prepend_bos=True,
        append_eos=True,
    )

    wordpiece_corpus = WordpieceCorpus(corpus, prepend_cls=True)

    import ipdb

    ipdb.set_trace()
