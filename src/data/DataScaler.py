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


energy_indices = [0, 1, 2, 3, 4, 5, 6]  # Assuming these are the indices for energy features
eta_indices = [7, 8, 9, 10, 11, 12, 13]  # Assuming these are the indices for eta features
phi_indices = [14, 15, 16, 17, 18, 19, 20]  # Assuming these are the indices for phi features
pt_indices = [21, 22, 23, 24, 25, 26, 27]  # Assuming these are the indices for pt features
btag_indices = [28, 29, 30, 31, 32, 33, 34]  # Assuming these are the indices for b-tag features
DNNScaler = ColumnTransformer(
    [('E', LogScaler(add_mask=True), energy_indices),
    ('phi', PhiTransformer(), phi_indices),
    ('eta',StandardScaler(), eta_indices),
    ('pt',LogScaler(), pt_indices),
    ('btag', MinMaxScaler(), btag_indices)
    ],
    remainder='passthrough'
)

if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")