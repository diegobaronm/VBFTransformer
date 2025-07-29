import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from loguru import logger

# In all these scalers its important to use functions that can work with NaNs and also not to 
# replace any of them with arbitrary numbers, as this is done in the pipeline later
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
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if self.add_mask:
            mask_features = [f"{feat}_mask" for feat in input_features]
            log_features = [f"log({feat})" for feat in input_features]
            return np.array(mask_features + log_features)
        else:
            return np.array([f"{feat}" for feat in input_features])

    def inverse_transform(self, X):
        if self.add_mask:
            # Assumes first half of features are masks, second half are log values
            X = X[:, X.shape[1] // 2:]
        return np.exp(X) - self.offset

class PhiTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        sin_phi = np.sin(X)
        cos_phi = np.cos(X)
        return np.hstack([sin_phi, cos_phi])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        # Extract the raw feature name without namespace (e.g., 'phi__x14' -> 'x14')
        def clean_name(name):
            if '__' in name:
                prefix, var = name.split('__', 1)
                return f"{prefix}{var}"  # e.g., phi__x14 -> phix14
            return name  # fallback

        out = []
        for feat in input_features:
            base = clean_name(feat)
            out.append(f"sin({base})")
            out.append(f"cos({base})")
        return np.array(out)

class ArctanScaler(FunctionTransformer):
    def __init__(self):
        super().__init__(func=lambda x: np.arctan(x) * 2 / np.pi,
                         inverse_func=lambda x: np.tan(x * np.pi / 2),
                         validate=True)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [f"{name}" for name in input_features]

class LogMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, add_mask=False):
        self.scaler = MinMaxScaler()
        self.add_mask = add_mask

    def fit(self, X, y=None):
        X_log = np.log1p(X)
        # Fit scaler only on non-NaN values (mask out NaNs)
        self.mask_ = np.isnan(X_log)
        # Use only non-NaN values for fitting scaler (flattened)
        self.scaler.fit(X_log[~self.mask_].reshape(-1, 1))
        return self

    def transform(self, X):
        X_log = np.log1p(X)
        mask = np.isnan(X_log)

        # Prepare array to store scaled results, initially all NaN
        X_scaled = np.full_like(X_log, np.nan, dtype=float)

        # Scale only non-NaN values
        non_nan_idx = ~mask
        # MinMaxScaler expects 2D array for transform
        scaled_values = self.scaler.transform(X_log[non_nan_idx].reshape(-1, 1)).flatten()
        X_scaled[non_nan_idx] = scaled_values

        if self.add_mask:
            return np.hstack([mask, X_scaled])
        return X_scaled

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        scaled_names = [f"{name}" for name in input_features]
        if self.add_mask:
            mask_names = [f"{name}_nan_mask" for name in input_features]
            return mask_names + scaled_names
        return scaled_names

    def inverse_transform(self, X):
        if self.add_mask:
            X = X[:, X.shape[1] // 2:]
        X_unscaled = self.scaler.inverse_transform(X.reshape(-1, 1)).flatten()
        return np.expm1(X_unscaled).reshape(X.shape)


        
class TanhScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return 0.5 * (np.tanh((X - self.mean_) / self.std_) + 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [f"{name}" for name in input_features]

    def inverse_transform(self, X):
        return self.std_ * np.arctanh(2 * X - 1) + self.mean_


# This scaler is needed when not applying scaling to data, if remainder='passthrough' is used the scaled data 
# is put before the un-scaled data hence shuffling the tensor and causing problems when plotting

class NoOpScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        return np.asarray(X)  # Return input unchanged

    def inverse_transform(self, X):
        return np.asarray(X)  # Also return unchanged

    def get_feature_names_out(self, input_features=None):
        return input_features

        
# DNN Scaler
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

# DNN Regression Scaler
scaler_map = {
    'standardscaler': StandardScaler,
    'minmax': MinMaxScaler,
    'logminmax': LogMinMaxScaler,
    'arctan': ArctanScaler,
    'tanh': TanhScaler,
    'log': LogScaler, 
    'none': NoOpScaler,
    'phitransformer': PhiTransformer,
}


def get_scalers_from_config(scaling_dict: dict) -> dict:
    scalers = {}
    if scaling_dict is None: return {}
        
    for feature, scaler_name in scaling_dict.items():
        scaler_cls = scaler_map.get(scaler_name.lower())
        if scaler_cls is not None:
            scalers[feature] = scaler_cls()  # instantiate the scaler
        else:
            scalers[feature] = None  # no scaling applied
    return scalers
    
# It gets fed an array of keywords to pull out and an array of indices to apply those transformations to
def create_custom_scaler(all_feature_indices, scaling_dict, extra_scaling_dict):
    transformers = []
    
    # Merge scaling configs
    full_scaling_dict = {**scaling_dict, **extra_scaling_dict}

    # Infer number of particles from one of the features
    num_particles = len(next(iter(all_feature_indices.values())))
    enable_masking = num_particles > 5

    if enable_masking:
        logger.info(f"[INFO] Particle number exceeds 5; enabling NaN mask for applicable scalers.")

    for feature_name, indices in all_feature_indices.items():
        scaler_name = full_scaling_dict.get(feature_name.lower())  # Normalize key

        scaler_cls = scaler_map.get(scaler_name.lower())
        if scaler_cls is None:
            raise ValueError(f"[ERROR] Unknown scaler type '{scaler_name}' for feature '{feature_name}'")

        # Special handling: inject add_mask=True for LogMinMaxScaler if particles > 5
        if scaler_cls.__name__.lower() == "logminmaxscaler" and enable_masking:
            scaler_instance = scaler_cls(add_mask=True)
            logger.info(f"[SCALING] Feature: '{feature_name}', Indices: {indices}, Scaler: {scaler_cls.__name__} (add_mask=True)")
            enable_masking=False # Only sensible to put the masks once
        else:
            scaler_instance = scaler_cls()
            logger.info(f"[SCALING] Feature: '{feature_name}', Indices: {indices}, Scaler: {scaler_cls.__name__}")

        transformers.append((feature_name, scaler_instance, indices))

    return ColumnTransformer(transformers, remainder='passthrough')


energy_indices = [0]

# Transformer Scaler
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