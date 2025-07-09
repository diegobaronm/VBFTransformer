import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1e-3, add_mask = False):
        self.offset = offset
        self.add_mask = add_mask

    def fit(self, X, y=None):
        return self  # no fitting needed

    def transform(self, X):
        log = np.log(X + self.offset)
        # Create a mask for nan values
        mask = np.isnan(log)
        if self.add_mask:
            return np.hstack([mask, log])
        return log

class PhiTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sin_phi = np.sin(X)
        cos_phi = np.cos(X)
        return np.hstack([sin_phi, cos_phi])

# DNN Scaler
energy_indices = np.array([0, 1, 2, 3, 4])  # Assuming these are the indices for energy features
eta_indices = energy_indices + 5 # Assuming these are the indices for eta features
phi_indices = energy_indices + 10 # Assuming these are the indices for phi features
pt_indices = energy_indices + 15 # Assuming these are the indices for pt features
btag_indices = energy_indices + 20 # Assuming these are the indices for b-tag features
DNNScaler = ColumnTransformer(
    [('E', LogScaler(add_mask=False), energy_indices),
    ('phi', PhiTransformer(), phi_indices),
    ('eta',StandardScaler(), eta_indices),
    ('pt',LogScaler(), pt_indices),
    ('btag', MinMaxScaler(), btag_indices)
    ],
    remainder='passthrough'
)

# Transformer Scaler
energy_indices = [0]
eta_indices = [1]
phi_indices = [2]
pt_indices = [3]
btag_indices = [4]
TransformerScaler = ColumnTransformer(
    [('E', LogScaler(add_mask=False), energy_indices),
    ('phi', PhiTransformer(), phi_indices),
    ('eta',StandardScaler(), eta_indices),
    ('pt',LogScaler(), pt_indices),
    ('btag', MinMaxScaler(), btag_indices)
    ],
    remainder='passthrough'
)

if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")