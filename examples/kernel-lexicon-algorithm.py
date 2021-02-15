import padasip as pa
import numpy as np
import matplotlib.pylab as plt

from torch.utils.data import DataLoader, SubsetRandomSampler

#from slp.data.diction import seeds_diction
from slp.util.embeddings import EmbeddingsLoader
from slp.data.transforms import SpacyTokenizer
from slp.data.therapy_lexicon import PsychologicalDataset, TupleDataset
from sklearn.metrics.pairwise import cosine_similarity

DATASET = '../../../whole-dataset.csv'


if __name__ == '__main__':
   
    Kseeds = 200
    max_word_length = 150
#    seed_set = list(seeds_diction.keys())

    loader = EmbeddingsLoader('/data/embeddings/glove.840B.300d.txt', 300)
    word2idx, idx2word, embeddings = loader.load()

    tokenizer = SpacyTokenizer()
    bio = PsychologicalDataset(
        DATASET,
        '../../../test_CEL/slp/data/psychotherapy',
        max_word_length,
        text_transforms = tokenizer)

    corpus = []
    for i, (text, title, lab) in enumerate(bio):
        corpus.extend(text)



    import pdb; pdb.set_trace()
    corpus = np.unique(corpus)
    vocabulary = [word for word in corpus if word not in seed_set]
    Nwords = len(vocabulary)

    #x-input matrix initialization
    x = np.zeros(Kseeds, Nwords)
    i = 0
    for word in vocabulary:
         wv = word2idx[word]
         j = 0
         for seed in seed_set:
            ws = word2idx[seed]
            d = cosine_similarity(wv, ws)
            x[i][j] = d * seeds_diction[seed]
            j += 1
         i += 1

    #filter definition
    f = pa.filters.FilterLMS(n=Nwords, mu=0.01, w="random")
    mul = np.matmul(x, d)
    f.run(mul, x)
