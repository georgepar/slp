import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, recall_score, precision_score, \
    accuracy_score, jaccard_similarity_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR, SVC

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

import slp.transforms.text.skl as text_transforms
from slp.transforms import SklIdentityTransformer, SklComposer


def eval_reg(y_hat, y):
    results = {
        "pearson": pearsonr([float(x) for x in y_hat],
                            [float(x) for x in y])[0]
    }

    return results


def eval_clf(y_test, y_p):
    results = {
        "f1": f1_score(y_test, y_p, average='macro'),
        "recall": recall_score(y_test, y_p, average='macro'),
        "precision": precision_score(y_test, y_p, average='macro'),
        "accuracy": accuracy_score(y_test, y_p)
    }

    return results


def eval_mclf(y, y_hat):
    results = {
        "jaccard": jaccard_similarity_score(np.array(y),
                                            np.array(y_hat)),
        "f1-macro": f1_score(np.array(y), np.array(y_hat),
                             average='macro'),
        "f1-micro": f1_score(np.array(y), np.array(y_hat),
                             average='micro')
    }

    return results


def bow_preprocessor(
        input='content', encoding='utf-8', decode_error='strict',
        strip_accents=None, lowercase=True, preprocessor=None,
        tokenizer='spacy', analyzer='word', stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), max_df=1.0,
        min_df=5, max_features=10000, vocabulary=None, binary=False,
        dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
        sublinear_tf=True, strip_punctuation=False):

    tokenizer_fn = SklIdentityTransformer().transform
    punct = SklIdentityTransformer()
    if strip_punctuation:
        punct = text_transforms.PunctuationStripper()

    if tokenizer is not None:
        if tokenizer == 'spacy':
            tokenizer_fn = text_transforms.SpacyTokenizer().transform
        elif tokenizer == 'nltk':
            tokenizer_fn = text_transforms.NltkTokenizer().transform
        else:
            tokenizer_fn = text_transforms.SplitTokenizer().transform

    feature_extractor =  TfidfVectorizer(
        input=input, encoding=encoding, decode_error=decode_error,
        strip_accents=strip_accents, lowercase=lowercase,
        preprocessor=preprocessor, tokenizer=tokenizer_fn, analyzer=analyzer,
        stop_words=stop_words, token_pattern=token_pattern,
        ngram_range=ngram_range, max_df=max_df, min_df=min_df,
        max_features=max_features, vocabulary=vocabulary, binary=binary,
        dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf)

    return SklComposer([punct, feature_extractor])


def nbow_preprocessor(embeddings, word2idx, aggregation='mean',
        lowercase=True, tokenizer=None, stopwords=True,
        strip_punctuation=False):
    tokenizer_fn = SklIdentityTransformer().transform
    punct = SklIdentityTransformer()
    if strip_punctuation:
        punct = text_transforms.PunctuationStripper()

    if tokenizer is not None:
        if tokenizer == 'spacy':
            tokenizer_fn = text_transforms.SpacyTokenizer().transform
        elif tokenizer == 'nltk':
            tokenizer_fn = text_transforms.NltkTokenizer().transform
        else:
            tokenizer_fn = text_transforms.SplitTokenizer().transform

    feature_extractor = text_transforms.NBOWVectorizer(
        embeddings, word2idx, aggregation=aggregation,
        lowercase=lowercase, tokenizer=tokenizer_fn, stopwords=stopwords)

    return SklComposer([punct, feature_extractor])



class BowClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self, input='content', encoding='utf-8', decode_error='strict',
            strip_accents=None, lowercase=True, preprocessor=None,
            tokenizer='spacy', analyzer='word', stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), max_df=1.0,
            min_df=5, max_features=10000, vocabulary=None, binary=False,
            dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
            sublinear_tf=True, strip_punctuation=False, classifier='lr',
            **classifier_params):
        clf = LogisticRegression(**classifier_params)
        if classifier == 'svm':
            clf = SVC(**classifier_params)

        preprocessor = bow_preprocessor(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf, strip_punctuation=strip_punctuation)

        self.scores_ = None

        self.model_ = Pipeline(
            [('preprocessor', preprocessor), ('classifier', clf)]
        )

    def fit(self, X, y):
        self.model_ = self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        self.scores_ = eval_clf(y, y_pred)
        return self.model_.score(y, y_pred)


class BowRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self, input='content', encoding='utf-8', decode_error='strict',
            strip_accents=None, lowercase=True, preprocessor=None,
            tokenizer='spacy', analyzer='word', stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), max_df=1.0,
            min_df=5, max_features=10000, vocabulary=None, binary=False,
            dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
            sublinear_tf=True, strip_punctuation=False, regressor='svr',
            **regressor_params):
        reg = SVR(**regressor_params)
        if regressor == 'linear':
            reg = LinearRegression(**regressor_params)

        preprocessor = bow_preprocessor(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf, strip_punctuation=strip_punctuation)

        self.scores_ = None

        self.model_ = Pipeline(
            [('preprocessor', preprocessor), ('regressor', reg)]
        )

    def fit(self, X, y):
        self.model_ = self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        self.scores_ = eval_reg(y, y_pred)
        return self.model_.score(y, y_pred)


class NbowClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embeddings, word2idx, aggregation='mean',
                 lowercase=True, tokenizer=None, stopwords=True,
                 strip_punctuation=False, classifier='lr',
                 **classifier_params):
        preprocessor = nbow_preprocessor(
            embeddings, word2idx, aggregation=aggregation, lowercase=lowercase,
            tokenizer=tokenizer, stopwords=stopwords,
            strip_punctuation=strip_punctuation)

        clf = LogisticRegression(**classifier_params)
        if classifier == 'svm':
            clf = SVC(**classifier_params)

        self.scores_ = None

        self.model_ = Pipeline(
            [('preprocessor', preprocessor), ('classifier', clf)]
        )

    def fit(self, X, y):
        self.model_ = self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        self.scores_ = eval_clf(y, y_pred)
        return self.model_.score(y, y_pred)