from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from scipy.stats import gaussian_kde
import numpy as np
import torch

import logging

from abc import ABC, abstractmethod

class SA:
    def __init__(self, mode='MLSA', dimension_reducer=None, n_modes=5):
        self.reference_set = None
        self.mode = mode

        self.ref_density_model = None
        self.ref_inv_cov = None
        self.ref_mean = None

        self.dimension_reducer = dimension_reducer
        self.n_modes = n_modes

    def register_reference_set(self, reference_set, contrast_set=None):
        if self.dimension_reducer is not None and not self.dimension_reducer._fitted:
            self.reference_set = self.dimension_reducer.fit(reference_set, contrast_set=contrast_set)
        
        self.reference_set = self.dimension_reducer.transform(reference_set)

        if self.mode == 'LSA' or self.mode == 'all':
            self.ref_density_model = gaussian_kde(self.reference_set.T, bw_method='scott')

        elif self.mode == 'MDSA' or self.mode == 'all':
            cov = torch.cov(self.reference_set.T)
            self.ref_inv_cov = torch.linalg.pinv(cov)
            self.ref_mean = torch.mean(self.reference_set, dim=0)

        elif self.mode == 'MLSA' or self.mode == 'all':
            self.ref_density_model = GaussianMixture(n_components=self.n_modes, random_state=0)
            self.ref_density_model.fit(self.reference_set)

    def calculate(self, target):
        if self.mode == 'all':
            return self.calculate_MLSA(target)
        
        elif self.mode == 'LSA':
            return self.calculate_LSA(target)
        
        elif self.mode == 'MDSA':
            return self.calculate_MDSA(target)

        elif self.mode == 'MLSA':
            return self.calculate_MLSA(target)

        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def calculate_LSA(self, target):
        if self.mode != 'LSA' and self.mode != 'all':
            raise ValueError("LSA is not available in the current mode: {}".format(self.mode))

        if self.dimension_reducer is not None:
            target_rep = self.dimension_reducer.transform(target)
        else:
            target_rep = target

        densities = self.ref_density_model.evaluate(target_rep.T)
        lsa_scores = -np.log(densities + 1e-30)
        return lsa_scores

    def calculate_MDSA(self, target):
        if self.mode != 'MDSA' and self.mode != 'all':
            raise ValueError("DSA is not available in the current mode: {}".format(self.mode))

        if self.dimension_reducer is not None:
            target_rep = self.dimension_reducer.transform(target)
        else:
            target_rep = target

        if self.ref_inv_cov is None or self.ref_mean is None:
            raise ValueError("Reference set has not been registered. Use `register_reference_set` method to register the reference set.")

        diff = target_rep - self.ref_mean
        left = torch.matmul(diff, self.ref_inv_cov)
        dist = torch.sqrt(torch.sum(left * diff, dim=1))
        return [dist.item() for dist in dist]

    def calculate_MLSA(self, target):
        if self.mode != 'MLSA' and self.mode != 'all':
            raise ValueError("MLSA is not available in the current mode: {}".format(self.mode))

        if self.dimension_reducer is not None:
            target_rep = self.dimension_reducer.transform(target)
        else:
            target_rep = target

        if self.ref_density_model is None:
            raise ValueError("Reference set has not been registered. Use `register_reference_set` method to register the reference set.")

        log_likelihood = self.ref_density_model.score_samples(target_rep)
        return -log_likelihood
