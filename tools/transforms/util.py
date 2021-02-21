from sklearn.base import BaseEstimator, TransformerMixin


class SklComposer(BaseEstimator, TransformerMixin):
    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, X, y=None):
        for trans in self.transforms:
            X = trans.transform(X)

        return X

    def fit(self, X, y=None):
        self.transforms = [t.fit(X, y) for t in self.transforms]
        return self


class SklIdentityTransformer(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return X

    def fit(self, X, y=None):
        return self
