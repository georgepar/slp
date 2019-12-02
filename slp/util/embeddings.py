import errno
import os

import numpy as np

from typing import cast, Any, Dict, Optional

from slp.config import SPECIAL_TOKENS
from slp.util import log
from slp.util import system
from slp.util import types


class EmbeddingsLoader(object):
    def __init__(self,
                 embeddings_file: str, dim: int,
                 extra_tokens: Any = SPECIAL_TOKENS) -> None:
        self.embeddings_file = embeddings_file
        self.cache_ = self._get_cache_name()
        self.dim_ = dim
        self.extra_tokens = extra_tokens

    def _get_cache_name(self) -> str:
        head, tail = os.path.split(self.embeddings_file)
        filename, ext = os.path.splitext(tail)
        cache_name = os.path.join(head, f'{filename}.p')
        log.info(f'Cache: {cache_name}')
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
            emb: Optional[np.ndarray] = None) -> types.Embeddings:
        word2idx[token] = len(embeddings)
        idx2word[len(embeddings)] = token
        if emb is None:
            emb = np.random.uniform(
                low=-0.05, high=0.05, size=self.dim_)
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
            log.info("Loaded word embeddings from cache.")
            return cache
        except OSError:
            log.warning(
                f"Didn't find embeddings cache file {self.embeddings_file}")

        # create the necessary dictionaries and the word embeddings matrix
        if not os.path.exists(self.embeddings_file):
            log.critical(f"{self.embeddings_file} not found!")
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT),
                          self.embeddings_file)

        log.info(f'Indexing file {self.embeddings_file} ...')

        # create the 2D array, which will be used for initializing
        # the Embedding layer of a NN.
        # We reserve the first row (idx=0), as the word embedding,
        # which will be used for zero padding (word with id = 0).
        word2idx, idx2word, embeddings = self.augment_embeddings(
            {}, {}, [], self.extra_tokens.PAD.value,
            emb=np.zeros(self.dim_))
        for token in self.extra_tokens:
            if token == self.extra_tokens.PAD:
                continue
            word2idx, idx2word, embeddings = self.augment_embeddings(
                word2idx, idx2word, embeddings, token.value)

        # read file, line by line
        with open(self.embeddings_file, "r") as f:
            index = len(embeddings)
            for line in f:
                # skip the first row if it is a header
                if len(line.split()) < self.dim_:
                    continue

                values = line.rstrip().split(" ")
                word = values[0]

                if word in word2idx:
                    continue

                vector = np.asarray(values[1:], dtype=np.float32)
                idx2word[index] = word
                word2idx[word] = index
                embeddings.append(vector)
                index += 1

        log.info(f'Found {len(embeddings)} word vectors.')
        embeddings = np.array(embeddings, dtype='float32')

        # write the data to a cache file
        self._dump_cache((word2idx, idx2word, embeddings))
        return word2idx, idx2word, embeddings


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        '../../cache/glove.840B.300d.txt', 300)
    embeddings = loader.load()
