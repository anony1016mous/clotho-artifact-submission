from sklearn.decomposition import PCA

from abc import ABC, abstractmethod
import numpy as np
import torch

class DimensionReducer(ABC):
    @abstractmethod
    def fit(self, ref_set):
        pass

    @abstractmethod
    def transform(self, targets):
        pass

    @abstractmethod
    def fit_transform(self, ref_set):
        pass


class PCAFeatureReducer(DimensionReducer):
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.pca = PCA(n_components=self.n_features)

    def fit(self, targets):
        self.pca.fit(targets)

    def transform(self, targets):
        return torch.tensor(self.pca.transform(targets))

    def fit_transform(self, targets):
        return torch.tensor(self.pca.fit_transform(targets))


class VarianceBasedFeatureSelector(DimensionReducer):
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.selected_dimensions = None
        
    def fit(self, targets):
        variances = np.var(targets, axis=0)
        self.selected_dimensions = np.argsort(variances)[-self.n_features:]
        
    def transform(self, targets):
        if self.selected_dimensions is None:
            raise ValueError("Feature selector has not been fitted yet. Call `fit` method first.")
        
        return targets[:, self.selected_dimensions]
    
    def fit_transform(self, targets):
        self.fit(targets)
        return self.transform(targets)
