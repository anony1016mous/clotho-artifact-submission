import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(42)

class PercentileRankClipper(BaseEstimator, TransformerMixin):
    def __init__(self, frac=0.05):
        self.frac = frac
        self.thresholds_ = None

    def fit(self, X, y=None):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        n, d = X_arr.shape
        cutoff_count = int(np.floor(self.frac * n))

        self.thresholds_ = np.empty(d, dtype=float)

        for j in range(d):
            col = X_arr[:, j]
            rank_index = min(cutoff_count, col.size) - 1
            self.thresholds_[j] = np.partition(col, rank_index)[rank_index]
        return self

    def transform(self, X):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        clipped = np.maximum(X_arr, self.thresholds_)
        if hasattr(X, "copy"):
            out = X.copy()
            out.loc[:, :] = clipped
            return out
        return clipped

def fit_logistic_model(X_train, y_train, scaler_type, clip_frac=0.05):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    model = Pipeline([
        ('clip', PercentileRankClipper(frac=clip_frac)),
        ('scaler', scaler),
        ('logreg', LogisticRegression(max_iter=3000, random_state=42))
    ])
    model.fit(X_train, y_train)
    return model

def predict_prob(model, X_test):
    return model.predict_proba(X_test)[:, 1]

def fit_single_feature_baseline(train_df, test_df, feature, threshold, scaler_type):
    X_train = train_df[[feature]]
    y_train = (train_df['test_score'].values >= threshold).astype(int)
    
    X_test = test_df[[feature]]
    
    model = fit_logistic_model(X_train, y_train, scaler_type)
    p_test = predict_prob(model, X_test)
    
    return p_test, model