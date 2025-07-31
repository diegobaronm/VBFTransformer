import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import polars as pl
import numpy as np
from loguru import logger
import h5py
from numpy.lib import recfunctions as rfn
from omegaconf import DictConfig
from enum import Enum  
from src.data.DataScaler import create_custom_scaler, scaler_map, get_scalers_from_config
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

from src.data.DataHelpers import read_h5_data, load_multiple_h5, flat_inputs, add_features, get_particle_feature_index_ranges, get_full_feature_index_ranges

from src.utils.Plotting import plot_particle_distributions
from src.utils.PrettyPrinting import prettify_feature_names

class MetadataIndex(Enum):
    # EVENT_NUMBER = 0 No need to assing this a n
    OMEGA = 1
    MZ_RECO = 2
    MZ_TRUTH = 3
    MMC_MZ = 4
    OPENING_ANGLE = 5
    LEP_MET_ANGLE_SIGNED = 6
    IS_MET_INSIDE = 7

particle_feature_dict = {
    'energy': 0,
    'eta': 1,
    'phi': 2,
    'pt': 3,
    'btag': 4,
    'charge': 5,
    'type': 6
}

extra_feature_dict = {
    'omega': MetadataIndex.OMEGA.value, 
    'mz_reco': MetadataIndex.MZ_RECO.value, 
    'mz_mmc': MetadataIndex.MMC_MZ.value, 
    'opening_angle': MetadataIndex.OPENING_ANGLE.value, 
    'lep_met_angle_signed': MetadataIndex.LEP_MET_ANGLE_SIGNED.value, 
    'is_met_inside': MetadataIndex.IS_MET_INSIDE.value, 
}

pretty_label_dict = {
    'energy': r"$E$",
    'eta': r"$\eta$",
    'pt': r"$p_{T}$",
    'phi': r"$\phi$",
    'btag': r"btag", 
    'charge': r"charge",
    'type': r"type",
    'phi__cos': r"$\cos(\phi)$",
    'phi__sin': r"$\sin(\phi)$",
}
    
extra_feature_label_dict = {
    'opening_angle':         r"$\Delta \phi_U(lep, tau)$",
    'lep_met_angle_signed': r"$\Delta \phi_S(lep, MET)$",
    'omega': r"$\Omega$", 
    'mz_reco': r"Mz-reco",
    'mz_mmc': r"Mz-mmc",    
}


class VBFDNNRegressionDataModule(L.LightningDataModule):
    def __init__(self, cfg_object: DictConfig):
        super().__init__()

        # === Constants === #
        MAX_PARTICLES = 10
        self.PEAK_RANGE = [88, 95]

        # === Dataset & model configuration === #
        dataset_cfg = cfg_object.dataset
        model_cfg = cfg_object.model
        train_cfg = cfg_object.train

        self.input_files = dataset_cfg.input_files
        self.train_batch_size = dataset_cfg.train.batch_size
        self.val_batch_size = dataset_cfg.val.batch_size
        self.train_num_workers = dataset_cfg.train.num_workers
        self.val_num_workers = dataset_cfg.val.num_workers

        if model_cfg.n_particles > MAX_PARTICLES: raise ValueError(f"n_particles must be ≤ {MAX_PARTICLES}")

        self.n_particles = model_cfg.n_particles
        self.features = [particle_feature_dict[k] for k in model_cfg.features]
        self.extra_features = [extra_feature_dict[k] for k in model_cfg.extra_features]

        self.num_quantiles = train_cfg.num_quantiles
        self.total_features = self.n_particles * len(self.features) + len(self.extra_features)

        # === Feature names === #
        self.particle_feature_names = model_cfg.features
        self.extra_feature_names = model_cfg.extra_features

        # === Scaling === #
        self.scaling_dict = dataset_cfg.scaling_dict
        self.extra_scaling_dict = dataset_cfg.extra_scaling_dict

        self.scalers = get_scalers_from_config(self.scaling_dict)
        self.extra_scaling_scalers = get_scalers_from_config(self.extra_scaling_dict)

        # === Result output directory === #
        self.result_dir = f'results/{model_cfg.name}/'

        # === Logging === #
        logger.info(f"Particle features: {self.particle_feature_names}")
        logger.info(f"Extra features: {self.extra_feature_names}")
        logger.info(f"Scaling dict: {self.scaling_dict}")
        logger.info(f"Scalers: {self.scalers}")
        logger.info(f"Extra scaling dict: {self.extra_scaling_dict}")
    
    def setup(self, stage: str):
        logger.info("Setting up the data module...")
    
        # === Load and prepare data === #
        inputs, metadata = self.__load_and_filter_data()
        input_data = self.__prepare_input_data(inputs, metadata)
        target = metadata[:, MetadataIndex.MZ_TRUTH.value]  # Monte Carlo truth mass
    
        logger.info(f"Particle Features inputted: {self.features}")
    
        # === Split data === #
        indices = np.arange(len(input_data))
        X_train, X_temp, y_train, y_temp, _, idx_temp = train_test_split( input_data, target, indices, test_size=0.2, random_state=0, shuffle=True )
        X_val, X_test, y_val, y_test, _, idx_test = train_test_split( X_temp, y_temp, idx_temp, test_size=0.5, shuffle=True, random_state=0 )
    
        # === Peak filtering and quantiles === #
        peak_indices = (y_train > self.PEAK_RANGE[0]) & (y_train < self.PEAK_RANGE[1])
        self.quantiles = np.quantile(y_train, np.linspace(0, 1, self.num_quantiles + 1))
    
        # === Human benchmark targets === #
        self.M_reco_human = metadata[:, MetadataIndex.MZ_RECO.value][idx_test]
        self.M_mmc_human  = metadata[:, MetadataIndex.MMC_MZ.value][idx_test]
    
        self.__plot_distributions(X_train, peak_indices, stage="Raw_Data", scaled=False) # Plot data before scaling
    
        all_feature_indices = get_full_feature_index_ranges(self.particle_feature_names, self.extra_feature_names, self.n_particles)
        scaler = create_custom_scaler(all_feature_indices, self.scaling_dict, self.extra_scaling_dict)
    
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)
    
        self.pretty_feature_names = prettify_feature_names(scaler.get_feature_names_out(), pretty_label_dict, extra_feature_label_dict, self.n_particles)
    
        self.__plot_distributions(X_train, peak_indices, stage="Scaled_Data", scaled=True) # Plot data after scaling
    
        # === Convert to tensors and dataset objects === #
        self.train_dataset = TensorDataset(self.__to_tensor(X_train), torch.tensor(y_train, dtype=torch.float32))
        self.val_dataset   = TensorDataset(self.__to_tensor(X_val),   torch.tensor(y_val,   dtype=torch.float32))
        self.test_dataset  = TensorDataset(self.__to_tensor(X_test),  torch.tensor(y_test,  dtype=torch.float32))
    
        # === Log sizes === #
        logger.info(f"Training dataset size: {len(y_train)}")
        logger.info(f"Validation dataset size: {len(y_val)}")
        logger.info(f"Test dataset size: {len(y_test)}")



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_num_workers, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, persistent_workers=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    # Private Class functions
    
    # Loads data from H5, safeguards against MMC failed events
    def __load_and_filter_data(self):
        signal_inputs, metadata_inputs, _ = load_multiple_h5(self.input_files)
        valid_mask = metadata_inputs[:, MetadataIndex.MMC_MZ.value] != 0
        return signal_inputs[valid_mask], metadata_inputs[valid_mask]


    def __prepare_input_data(self, particles, metadata):
        base = flat_inputs(particles, self.n_particles, self.features)
        extras = [metadata[:, i] for i in self.extra_features]
        return add_features(base, extras)

    def __to_tensor(self, X, NAN_PAD_VALUE=-1):
        return torch.nan_to_num(torch.tensor(X, dtype=torch.float32), nan=NAN_PAD_VALUE)

    def __plot_distributions(self, X, peak_mask, stage="Raw_Data", scaled=False):
        folder = self.result_dir + stage
        num_extra = len(self.extra_feature_names)
    
        if not scaled:
            # Plot particle features
            for idx, name in enumerate(self.features):
                start = idx * self.n_particles
                end = start + self.n_particles
                plot_particle_distributions(
                    peak_mask,
                    X[:, start:end],
                    title=self.particle_feature_names[idx],
                    n_bins=40,
                    folder_name=folder
                )
    
            # Plot extra features
            last_index = len(self.features) * self.n_particles
            for idx, name in enumerate(self.extra_feature_names):
                plot_particle_distributions(
                    peak_mask,
                    X[:, last_index + idx : last_index + idx + 1],
                    title=name,
                    n_bins=40,
                    no_particle_name=True,
                    folder_name=folder
                )
        else:
            # Plot scaled particle features in bundles
            for start_idx in range(0, len(self.pretty_feature_names) - num_extra, self.n_particles):
                end_idx = min(start_idx + self.n_particles, len(self.pretty_feature_names) - num_extra)
                data_bundle = X[:, start_idx:end_idx]
                title_bundle = self.pretty_feature_names[start_idx:end_idx]
                plot_particle_distributions(
                    peak_mask,
                    data_bundle,
                    title=f"Features {start_idx} to {end_idx - 1}",
                    titles=title_bundle,
                    n_bins=40,
                    folder_name=folder
                )
    
            # Plot scaled extra features
            for i in range(len(self.pretty_feature_names) - num_extra, len(self.pretty_feature_names)):
                data_extra = X[:, i].reshape(-1, 1)
                plot_particle_distributions(
                    peak_mask,
                    data_extra,
                    title=self.pretty_feature_names[i],
                    n_bins=40,
                    no_particle_name=True,
                    folder_name=folder
                )
    
    
    
    
        
        
