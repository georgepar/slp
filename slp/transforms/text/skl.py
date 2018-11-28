import nltk
import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

import slp.transforms.text.functional as functional


class Untokenizer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return [functional.untokenize(x) for x in X]

    def fit(self, X, y=None):
        return self


class PunctuationStripper(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return [functional.strip_punctuation(s) for s in X]

    def fit(self, X, y=None):
        return self


class SplitTokenizer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return [functional.split_tokenizer(s) for s in X]

    def fit(self, X, y=None):
        return self


class NltkTokenizer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return [nltk.word_tokenize(s) for s in X]

    def fit(self, X, y=None):
        return self


class SpacyTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang='en'):
        self.nlp = spacy.load(lang)

    def transform(self, X, y=None):
        return [self.nlp.tokenizer(s) for s in X]

    def fit(self, X, y=None):
        return self


class NBOWVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 embeddings,
                 word2idx,
                 aggregation='mean',
                 lowercase=True,
                 tokenizer=None,
                 stopwords=True):
        self.aggregation = aggregation
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.dim = embeddings[0].size
        self.stopwords = stopwords
        self.stops = set(nltk.corpus.stopwords.words('english'))
        self.lowercase = lowercase
        if tokenizer is not None:
            self.tokenizer = lambda x: x
        else:
            self.tokenizer = tokenizer

    def transform(self, X, y=None):
        docs = []
        for doc in X:
            vectors = []
            if self.lowercase:
                doc = doc.lower()
            for word in self.tokenizer(doc):
                if word not in self.word2idx:
                    continue
                if not self.stopwords and word in self.stops:
                    continue
                vectors.append(self.embeddings[self.word2idx[word]])
            if not vectors:
                vectors.append(np.zeros(self.dim))
            feats = functional.aggregate_vecs(np.array(vectors),
                                              aggregation=self.aggregation)
            docs.append(feats)

        assert len(docs) == X.shape[0]
        return check_array(docs)

    def fit(self, X, y=None):
        return self
