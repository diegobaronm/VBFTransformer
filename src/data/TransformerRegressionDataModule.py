import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset, WeightedRandomSampler
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
from itertools import combinations
from src.data.DataHelpers import read_h5_data, load_multiple_h5, flat_inputs, add_features, get_particle_feature_index_ranges, get_full_feature_index_ranges, calculate_KDE_sampler, calculate_interaction_inputs

from src.utils.Plotting import plot_particle_distributions, plot_kde_and_inverse_weights, plot_resampled_distributions
from src.utils.PrettyPrinting import prettify_feature_names

from src.data.DataFormat import MetadataIndex, particle_feature_dict, extra_feature_dict, pretty_label_dict, extra_feature_label_dict
        
class VBFTransformerRegressionDataModule(L.LightningDataModule):
    def __init__(self, cfg_object: DictConfig):
        super().__init__()

        # === Constants === #
        self.MAX_PARTICLES = 10
        self.PEAK_RANGE = [91.8 - 5, 91.8 + 5] # +- 5 GeV from Z-peak, Splits data in 3 regions

        # === Dataset & model configuration === #
        dataset_cfg, model_cfg, train_cfg = cfg_object.dataset, cfg_object.model, cfg_object.train

        self.input_files = dataset_cfg.input_files
        self.train_batch_size = dataset_cfg.train.batch_size
        self.val_batch_size = dataset_cfg.val.batch_size
        self.train_num_workers = dataset_cfg.train.num_workers
        self.val_num_workers = dataset_cfg.val.num_workers

        self.n_particles = model_cfg.n_particles
        self.features = [particle_feature_dict[k] for k in dataset_cfg.scaling_dict.keys()]
        self.extra_features = [extra_feature_dict[k] for k in dataset_cfg.extra_scaling_dict.keys()]

        # Once you want to get the interaction transformer in the paper
        self.compute_pairing_tokens = train_cfg.get('compute_interaction_tokens', False)
        self.inverse_sampling = dataset_cfg.get('inverse_sampling', False)

        self.KDE_width = train_cfg.get('KDE_width', 2)
        self.Min_Dens_cap = train_cfg.get('Min_Dens_cap', 10)
    
        self.num_quantiles = train_cfg.num_quantiles
        self.total_features = self.n_particles * len(self.features) + len(self.extra_features)

        # === Feature names === #
        self.particle_feature_names = np.array(list(dataset_cfg.scaling_dict.keys()))
        self.extra_feature_names = np.array(list(dataset_cfg.extra_scaling_dict.keys())) 

        # === Scaling === #
        self.scaling_dict = dataset_cfg.scaling_dict
        self.extra_scaling_dict = dataset_cfg.extra_scaling_dict

        self.scalers = get_scalers_from_config(self.scaling_dict)
        self.extra_scaling_scalers = get_scalers_from_config(self.extra_scaling_dict)
        self.target_scaler = scaler_map[dataset_cfg.target_scaling]()

        # === Result output directory === #
        self.result_dir = f'results/{model_cfg.name}/'

        self.using_cross_attention = len(self.extra_features) > 0
        logger.info(f"Using cross attention: {self.using_cross_attention or self.compute_pairing_tokens}")

        # === Logging === #
        logger.info(f"Particle features: {self.particle_feature_names}")
        logger.info(f"Extra features: {self.extra_feature_names}")
        logger.info(f"Scaling dict: {self.scaling_dict}")
        logger.info(f"Scalers: {self.scalers}")
        logger.info(f"Extra scaling dict: {self.extra_scaling_dict}")
        logger.info(f"Scaling of the targets: {self.target_scaler}")
    
    def setup(self, stage: str):
        if self.n_particles > self.MAX_PARTICLES: raise ValueError(f"n_particles must be ≤ {self.MAX_PARTICLES}")
        
        logger.info("Setting up the data module...")
    
        # === Load and prepare data === #
        inputs, metadata = self.__load_and_filter_data()

        if self.compute_pairing_tokens:
            interaction_tokens, pair_indices = calculate_interaction_inputs(inputs, self.n_particles)
            logger.info(f"Shape of the interaction tokens: {interaction_tokens.shape}")

        target = metadata[:, MetadataIndex.MZ_TRUTH.value]  # Monte Carlo truth mass

        input_data, metadata_data = self.__prepare_input_data(inputs, metadata)
        metadata_data = interaction_tokens if self.compute_pairing_tokens else metadata_data.T

        logger.info(metadata_data.shape)
        logger.info(f"Particle Features inputted: {self.features}")
    
        # === Split data === #
        indices = np.arange(len(input_data))
        
        if self.using_cross_attention or self.compute_pairing_tokens:
            
            if self.compute_pairing_tokens: 
                metadata_data = interaction_tokens

                
            # First split: 80% train, 20% temp (val + test)
            (p_train, p_temp,
             m_train, m_temp,
             y_train, y_temp,
             idx_train, idx_temp) = train_test_split(
                input_data, metadata_data, target, indices,
                test_size=0.2, random_state=0, shuffle=True
            )
            
            # Second split: 10% val, 10% test
            (p_val, p_test,
             m_val, m_test,
             y_val, y_test,
             idx_val, idx_test) = train_test_split(
                p_temp, m_temp, y_temp, idx_temp,
                test_size=0.5, random_state=0, shuffle=True
            )

        else:
            # First split: 80% train, 20% temp (val + test)
            (p_train, p_temp,
             y_train, y_temp,
             idx_train, idx_temp) = train_test_split(
                input_data, target, indices,
                test_size=0.2, random_state=0, shuffle=True
            )
            
            # Second split: 10% val, 10% test
            (p_val, p_test,
             y_val, y_test,
             idx_val, idx_test) = train_test_split(
                p_temp, y_temp, idx_temp,
                test_size=0.5, random_state=0, shuffle=True
            )


        self.orignal_y_train = y_train
        
        # === Peak filtering and quantiles === #
        left_tail_ind = y_train < self.PEAK_RANGE[0]
        right_tail_ind = y_train > self.PEAK_RANGE[1]
        
        self.quantiles = np.quantile(y_train, np.linspace(0, 1, self.num_quantiles + 1))
    
        # === Human benchmark targets === #
        self.M_reco_human = metadata[:, MetadataIndex.MZ_RECO.value][idx_test]
        self.M_mmc_human  = metadata[:, MetadataIndex.MMC_MZ.value][idx_test]

        if not self.using_cross_attention:
            self.__plot_distributions(p_train, [], left_tail_ind, right_tail_ind, stage="Raw_Data", scaled=False) # Plot data before scaling
        else: 
            self.__plot_distributions(p_train, m_train, left_tail_ind, right_tail_ind, stage="Raw_Data", scaled=False) # Plot data after scaling
        
        plot_particle_distributions(
                    left_tail_ind, right_tail_ind,
                    y_train,
                    title='Target_feature',
                    n_bins=40,
                    no_particle_name=True,
                    folder_name= self.result_dir + 'Raw_Data'
                )

        if not self.using_cross_attention:
            all_feature_indices = get_full_feature_index_ranges(self.particle_feature_names, self.extra_feature_names, self.n_particles)
        else:
            all_feature_indices = get_full_feature_index_ranges(self.particle_feature_names, {}, self.n_particles)
            m_feature_indices = get_full_feature_index_ranges({}, self.extra_feature_names, self.n_particles)
            logger.info(f"Extra feature indices: {m_feature_indices}")
            
        if not self.using_cross_attention:
            p_scaler = create_custom_scaler(all_feature_indices, self.scaling_dict, self.extra_scaling_dict)
        else: 
            p_scaler = create_custom_scaler(all_feature_indices, self.scaling_dict, {})
            m_scaler = create_custom_scaler(m_feature_indices, {}, self.extra_scaling_dict)
    
        p_train = p_scaler.fit_transform(p_train)
        p_val   = p_scaler.transform(p_val)
        p_test  = p_scaler.transform(p_test)
        
        if self.using_cross_attention: 
            m_train = m_scaler.fit_transform(m_train)
            m_val = m_scaler.fit_transform(m_val)
            m_test = m_scaler.fit_transform(m_test)

        # Now do the inverse scaling and change the distribution of y_train: 
        if self.inverse_sampling: 
            logger.info(f'Creating a KDE sampler')
            self.density, self.sampler = calculate_KDE_sampler(y_train, KDE_width=self.KDE_width, Min_Dens_cap=self.Min_Dens_cap)
        
        y_train = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).T[0]
        y_val   = self.target_scaler.transform(y_val.reshape(-1, 1)).T[0]
        y_test  = self.target_scaler.transform(y_test.reshape(-1, 1)).T[0]

        self.pretty_feature_names = prettify_feature_names(p_scaler.get_feature_names_out(), pretty_label_dict, extra_feature_label_dict, self.n_particles)

        if self.using_cross_attention: 
            joined_features = np.append(p_scaler.get_feature_names_out(), m_scaler.get_feature_names_out())
            self.pretty_feature_names = prettify_feature_names(joined_features, pretty_label_dict, extra_feature_label_dict, self.n_particles)

        if not self.using_cross_attention:
            self.__plot_distributions(p_train, [], left_tail_ind, right_tail_ind, stage="Scaled_Data", scaled=True) # Plot data after scaling
        else: 
            self.__plot_distributions(p_train, m_train, left_tail_ind, right_tail_ind, stage="Scaled_Data", scaled=True) # Plot data after scaling
            

        plot_particle_distributions(
                    left_tail_ind, right_tail_ind,
                    y_train,
                    title='Target_feature',
                    n_bins=40,
                    no_particle_name=True,
                    folder_name= self.result_dir + 'Scaled_Data'
                )

        # Reshape post plotting for training
        self.input_dim  = p_train.shape[1] // self.n_particles 
        if self.compute_pairing_tokens:
            self.event_dim  = m_train.shape[2]
        
        p_train = p_train.reshape(p_train.shape[0], self.input_dim, -1)
        p_val   = p_val.reshape(p_val.shape[0], self.input_dim, -1)
        p_test  = p_test.reshape(p_test.shape[0], self.input_dim, -1)

        # You always have to transpose the last two dimensions
        p_train = p_train.transpose(0, 2, 1)
        p_val = p_val.transpose(0, 2, 1)
        p_test = p_test.transpose(0, 2, 1)

        if self.trainer.datamodule.using_cross_attention or self.compute_pairing_tokens:
            print(f'Shape of interaction training tokens: {m_train.shape}')
        print(f'Shape of particle training tokens: {p_train.shape}')

        if self.trainer.datamodule.using_cross_attention or self.compute_pairing_tokens:
            logger.info("Creating a dataset with metada")
            self.train_dataset = TensorDataset(self.__to_tensor(p_train), self.__to_tensor(m_train), torch.tensor(y_train, dtype=torch.float32))
            self.val_dataset   = TensorDataset(self.__to_tensor(p_val), self.__to_tensor(m_val),   torch.tensor(y_val,   dtype=torch.float32))
            self.test_dataset  = TensorDataset(self.__to_tensor(p_test), self.__to_tensor(m_test),  torch.tensor(y_test,  dtype=torch.float32))
            
        else: 
            # === Convert to tensors and dataset objects === #
            self.train_dataset = TensorDataset(self.__to_tensor(p_train), torch.tensor(y_train, dtype=torch.float32))
            self.val_dataset   = TensorDataset(self.__to_tensor(p_val),   torch.tensor(y_val,   dtype=torch.float32))
            self.test_dataset  = TensorDataset(self.__to_tensor(p_test),  torch.tensor(y_test,  dtype=torch.float32))
    
        # === Log sizes === #
        logger.info(f"The length of each token is: {self.input_dim}")
        logger.info(f"Training dataset size: {len(y_train)}")
        logger.info(f"Validation dataset size: {len(y_val)}")
        logger.info(f"Test dataset size: {len(y_test)}")


    def train_dataloader(self):
        
        if self.inverse_sampling:
            data_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                sampler=self.sampler,  # <- use sampler
                num_workers=15,
                persistent_workers=True,
                drop_last=True,
                pin_memory=True
            )

            
            weights = self.sampler.weights.numpy()
            
            y_vals = []

            if self.compute_pairing_tokens: 
                for i, (_, _, y_batch) in enumerate(data_loader):
                    y_vals.extend(y_batch.cpu().numpy())
        
            else: 
                for i, (_, y_batch) in enumerate(data_loader):
                    y_vals.extend(y_batch.cpu().numpy())
            
            y_vals = self.target_scaler.inverse_transform(np.array(y_vals).reshape(-1,1)).flatten()

            plot_kde_and_inverse_weights(
                self.orignal_y_train,
                self.density,
                np.minimum(1.0 / (self.density + 1e-8), self.Min_Dens_cap),
                save_path=self.result_dir + '/KDE_weights.png',
                xlim=(60, 120)
            )
            
            plot_resampled_distributions(self.orignal_y_train, y_vals, self.result_dir + '/KDE.png')

            logger.info(f"Finished plotting re-sampled distribution")
            
            return data_loader

        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_num_workers, persistent_workers=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, persistent_workers=True, drop_last=True, pin_memory=True)

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
        extras = np.array([metadata[:, i] for i in self.extra_features])
        return base, extras

    def __to_tensor(self, X, NAN_PAD_VALUE=-1):
        return torch.nan_to_num(torch.tensor(X, dtype=torch.float32), nan=NAN_PAD_VALUE)

    def __plot_distributions(self, p_data, m_data, left_tail_ind, right_tail_ind, stage="Raw_Data", scaled=False):
        folder = self.result_dir + stage
        num_extra = len(self.extra_feature_names)
    
        if not scaled:
            # Plot particle features
            for idx, name in enumerate(self.features):
                start = idx * self.n_particles
                end = start + self.n_particles
                plot_particle_distributions(
                    left_tail_ind, right_tail_ind,
                    p_data[:, start:end],
                    title=self.particle_feature_names[idx],
                    n_bins=40,
                    folder_name=folder
                )
    
            # Plot extra features
            for idx, name in enumerate(self.extra_feature_names):
                data_extra = m_data[:, idx].reshape(-1, 1)
                plot_particle_distributions(
                    left_tail_ind, right_tail_ind,
                    m_data[:, idx],
                    title=name,
                    n_bins=40,
                    no_particle_name=True,
                    folder_name=folder
                )
        else:
            # Plot scaled particle features in bundles
            for start_idx in range(0, len(self.pretty_feature_names) - num_extra, self.n_particles):
                end_idx = min(start_idx + self.n_particles, len(self.pretty_feature_names) - num_extra)
                data_bundle = p_data[:, start_idx:end_idx]
                title_bundle = self.pretty_feature_names[start_idx:end_idx]
                plot_particle_distributions(
                    left_tail_ind, right_tail_ind,
                    data_bundle,
                    title=f"Features {start_idx} to {end_idx - 1}",
                    titles=title_bundle,
                    n_bins=40,
                    folder_name=folder
                )
    
            # Plot scaled extra features
            for i in range(len(self.extra_feature_names)):
                data_extra = m_data[:, i].reshape(-1, 1)
                plot_particle_distributions(
                    left_tail_ind, right_tail_ind,
                    data_extra,
                    title=self.extra_feature_names[i],
                    n_bins=40,
                    no_particle_name=True,
                    folder_name=folder
                )
                