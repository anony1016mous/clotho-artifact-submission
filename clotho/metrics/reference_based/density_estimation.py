
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.utils.validation import check_is_fitted
from scipy.stats import entropy, rankdata
import numpy as np

from abc import ABC, abstractmethod
import logging


class DensityEstimator(ABC):
    @abstractmethod
    def fit(self, ref_set, labels=None):
        """Fit the density estimator to the reference set."""
        pass
    
    @abstractmethod
    def score(self, target_vectors):
        """Compute the density score for the target vectors."""
        pass
    
    @abstractmethod
    def rank(self, target_vectors):
        """Rank the target vectors based on their density scores."""
        pass


class GMMScorer(DensityEstimator):
    def __init__(self, n_components=10, random_state=42, warm_start=False, verbose=False, max_iter=100, reg_covar=1e-3, tol=1e-3, n_init=1, use_custom_class=False):
        if verbose:
            verbose_option = 2
            verbose_interval = 1
        else:
            verbose_option = 0
            verbose_interval = 1
            
        if use_custom_class:
            raise NotImplementedError("Custom GaussianMixture class is not implemented.")
        else:
            self.mixture_model_class = GaussianMixture

        self.model = self.mixture_model_class(n_components=n_components, random_state=random_state, warm_start=warm_start, verbose=verbose_option, verbose_interval=verbose_interval, max_iter=max_iter, reg_covar=reg_covar, tol=tol, n_init=n_init)
        self.n_components = n_components

    def fit(self, ref_set, labels=None):
        """Fit the GMM to the embeddings of the reference test cases. The embeddings should be reduced to a lower dimension."""
        sample_weights = None
        if labels is None:
            _ref_set_effective = ref_set
        elif all([label in [0, 1] for label in labels]):
            _ref_set_effective = ref_set[labels == 1]
        else:
            raise ValueError("Labels should be binary (0 or 1) if provided.")
            
        self.model.fit(_ref_set_effective)

    def score(self, target_vectors):
        """Compute the density score for the target vectors."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        logprobs = self.model.score_samples(target_vectors)
        return logprobs
    
    def rank(self, target_vectors, normalize=True):
        """Rank the target vectors based on their density scores."""
        scores = self.score(target_vectors)
        ranks = rankdata(scores, method='average')
        if normalize:
            ranks = (ranks - 1) / (len(ranks) - 1)  # Normalize ranks to [0, 1]
        return ranks
    
    def check_is_fitted(self):
        try:
            check_is_fitted(self.model)
        except Exception:
            return False
        
        return True

    def _n_features(self):
        return self.model.n_features_in_
    
    def _n_components(self):
        return self.model.n_components
    
    def get_uncertainties(self, target_vectors):
        """Get uncertainties for the target vectors based on the model's component assignment entropy."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        probs = self.model.predict_proba(target_vectors)
        logging.info(f"Component assignment probabilities stats: mean={np.mean(probs):.4f}, std={np.std(probs):.4f}, min={np.min(probs):.4f}, max={np.max(probs):.4f}")
        logging.info(f"Component assignment probabilities: {probs}")
        return entropy(probs, axis=1)
