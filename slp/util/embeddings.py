import errno
import os

import numpy as np

import slp.util.system as sys_util
import slp.util.log as log


class EmbeddingsLoader(object):
    def __init__(self, embeddings_file, dim,
                 extra_tokens=['<pad>', '<unk>', '<mask>', '<bos>', '<eos>']):
        self.logger = log.getLogger(f'{__name__}.EmbeddingsLoader')
        self.embeddings_file = embeddings_file
        self.cache_ = self._get_cache_name()
        self.dim_ = dim

    def _get_cache_name(self):
        head, tail = os.path.split(self.embeddings_file)
        filename, ext = os.path.splitext(tail)
        cache_name = os.path.join(head, f'{filename}.p')
        self.logger.info(f'Cache: {cache_name}')
        return cache_name

    def _dump_cache(self, data):
        sys_util.pickle_dump(data, self.cache_)

    def _load_cache(self):
        return sys_util.pickle_load(self.cache_)

    def load(self):
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
            self.logger.info("Loaded word embeddings from cache.")
            return cache
        except OSError:
            self.logger.warning(
                f"Didn't find embeddings cache file {self.embeddings_file}")

        # create the necessary dictionaries and the word embeddings matrix
        if not os.path.exists(self.embeddings_file):
            self.logger.critical(f"{self.embeddings_file} not found!")
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT),
                          self.embeddings_file)

        self.logger.info(f'Indexing file {self.embeddings_file} ...')

        word2idx = {}  # dictionary of words to ids
        idx2word = {}  # dictionary of ids to words
        embeddings = []  # the word embeddings matrix

        # create the 2D array, which will be used for initializing
        # the Embedding layer of a NN.
        # We reserve the first row (idx=0), as the word embedding,
        # which will be used for zero padding (word with id = 0).
        embeddings.append(np.zeros(self.dim_))

        # flag indicating whether the first row of the embeddings file
        # has a header
        header = False

        # read file, line by line
        with open(self.embeddings_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):

                # skip the first row if it is a header
                if i == 1:
                    if len(line.split()) < self.dim_:
                        header = True
                        continue

                values = line.split(" ")
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')

                index = i - 1 if header else i

                idx2word[index] = word
                word2idx[word] = index
                embeddings.append(vector)

            # add an unk token, for OOV words
            if "<unk>" not in word2idx:
                idx2word[len(idx2word) + 1] = "<unk>"
                word2idx["<unk>"] = len(word2idx) + 1
                embeddings.append(np.random.uniform(
                    low=-0.05, high=0.05, size=self.dim_))

            self.logger.info(f'Found {len(embeddings)} word vectors.')
            embeddings = np.array(embeddings, dtype='float32')

        # write the data to a cache file
        self._dump_cache((word2idx, idx2word, embeddings))

        return word2idx, idx2word, embeddings


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        '/home/geopar/projects/VQA/data/glove/glove.6B.300d.txt', 300)
    embeddings = loader.load()
