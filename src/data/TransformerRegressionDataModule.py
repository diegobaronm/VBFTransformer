# === Standard library === #
import os
import sys
from enum import Enum
from itertools import combinations

# === Third-party libraries === #
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

# === Project imports === #
from src.data.DataScaler import create_custom_scaler, scaler_map, get_scalers_from_config
from src.data.DataHelpers import prepare_input_data, get_full_feature_index_ranges, calculate_KDE_sampler, calculate_interaction_inputs, load_and_filter_data, to_float_tensor
from src.data.DataFormat import MetadataIndex, particle_feature_dict, extra_feature_dict, pretty_label_dict, extra_feature_label_dict
from src.utils.Plotting import plot_particle_distributions, plot_kde_and_inverse_weights, plot_resampled_distributions
from src.utils.PrettyPrinting import prettify_feature_names
        
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

        # If extra features defined in: 10.1088/1674-1137/ad7f3d should be computed
        self.compute_pairing_tokens = train_cfg.get('compute_pairing_tokens', False)
        self.using_extra_features = len(self.extra_features) > 0 
        
        self.inverse_sampling = dataset_cfg.get('inverse_sampling', False)
        if self.inverse_sampling:
            self.KDE_width = dataset_cfg.KDE_width
            self.Min_Dens_cap = dataset_cfg.Min_Dens_cap
    
        self.num_quantiles = train_cfg.num_quantiles

        # === Feature names and scaling === #
        self.particle_feature_names = np.array(list(dataset_cfg.scaling_dict.keys()))
        self.extra_feature_names = np.array(list(dataset_cfg.extra_scaling_dict.keys())) 

        self.scaling_dict = dataset_cfg.scaling_dict
        self.extra_scaling_dict = dataset_cfg.extra_scaling_dict

        self.scalers = get_scalers_from_config(self.scaling_dict)
        self.extra_scaling_scalers = get_scalers_from_config(self.extra_scaling_dict)
        self.target_scaler = scaler_map[dataset_cfg.target_scaling]()

        # === Result output directory === #
        self.result_dir = f'results/{model_cfg.name}/'

        # === Logging === #
        logger.info(f"Particle features: {self.particle_feature_names}")
        logger.info(f"Extra features: {self.extra_feature_names}")
        logger.info(f"Scaling dict: {self.scaling_dict}")
        logger.info(f"Scalers: {self.scalers}")
        logger.info(f"Extra scaling dict: {self.extra_scaling_dict}")
        logger.info(f"Scaling of the targets: {self.target_scaler}")

        # Private variable to ensure set up is not done twice when calling training and test scripts back to back
        self._has_setup = False
    
    def setup(self, stage: str):
        if self._has_setup:
            return
        self._has_setup = True

        if self.n_particles > self.MAX_PARTICLES: raise ValueError(f"n_particles must be ≤ {self.MAX_PARTICLES}")
        if self.using_extra_features and self.compute_pairing_tokens: raise ValueError(f"Cannot use both extra input features in this version")
        
        logger.info("Setting up the data module...")
    
        # === Load and prepare data === #
        inputs, metadata = load_and_filter_data(self.input_files)
        target = metadata[:, MetadataIndex.MZ_TRUTH.value]  # Monte Carlo truth mass

        input_data, metadata_data = prepare_input_data(inputs, metadata, self.n_particles, self.features, self.extra_features, combine=False)

        # We use metadata_data as a multi-purpose array, if we have interaction tokens we store those, if we have extra features we store those
        # if we have neither, we fill it with zeros and just avoid including it at the end, this allows to use one pipeline for data manipulation.
        if self.compute_pairing_tokens:
            metadata_data, pair_indices = calculate_interaction_inputs(inputs, self.n_particles)
        elif self.using_extra_features:
            metadata_data = metadata_data.T 
        else:
            metadata_data = np.zeros((len(inputs), 0))

        logger.info(f"Shape of metadata: {metadata_data.shape}")
        logger.info(f"Particle Features inputted: {self.features}")
    
        # === Split data (80 % Train, 10% val, 10% test) === # 
        indices = np.arange(len(input_data))
        p_train, p_temp, m_train, m_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(input_data, metadata_data, target, indices,
                                                                                                   test_size=0.2, random_state=0, shuffle=True)
        p_val, p_test, m_val, m_test, y_val, y_test, idx_val, idx_test = train_test_split(p_temp, m_temp, y_temp, idx_temp, 
                                                                                          test_size=0.5, random_state=0, shuffle=True)
        # === Peak filtering and quantiles === #
        left_tail_ind = y_train < self.PEAK_RANGE[0]
        right_tail_ind = y_train > self.PEAK_RANGE[1]
        
        self.quantiles = np.quantile(y_train, np.linspace(0, 1, self.num_quantiles + 1))

        # Again because of the dummy m_train variable, unless we are using extra features, no plotting function with m_train will be called as
        # the labels for those plots dont exists as len(self.extra_features) is zero
        self.__plot_distributions(p_train, m_train, left_tail_ind, right_tail_ind, stage="Raw_Data", scaled=False) # Plot data after scaling
        
        plot_particle_distributions(left_tail_ind, right_tail_ind, y_train, title='Target_feature', n_bins=40, no_particle_name=True, 
                                    folder_name= self.result_dir + 'Raw_Data')

        if not self.using_extra_features:
            all_feature_indices = get_full_feature_index_ranges(self.particle_feature_names, self.extra_feature_names, self.n_particles)
            p_scaler = create_custom_scaler(all_feature_indices, self.scaling_dict, self.extra_scaling_dict)

        else:
            all_feature_indices = get_full_feature_index_ranges(self.particle_feature_names, {}, self.n_particles)
            m_feature_indices = get_full_feature_index_ranges({}, self.extra_feature_names, self.n_particles)

            p_scaler = create_custom_scaler(all_feature_indices, self.scaling_dict, {})
            m_scaler = create_custom_scaler(m_feature_indices, {}, self.extra_scaling_dict)
            
            logger.info(f"Extra feature indices: {m_feature_indices}")
    
        p_train = p_scaler.fit_transform(p_train)
        p_val   = p_scaler.transform(p_val)
        p_test  = p_scaler.transform(p_test)
        
        if self.using_extra_features: 
            m_train = m_scaler.fit_transform(m_train)
            m_val = m_scaler.fit_transform(m_val)
            m_test = m_scaler.fit_transform(m_test)

        # Now do the inverse scaling and change the distribution of y_train: 
        if self.inverse_sampling: 
            logger.info(f'Creating a KDE sampler')
            self.density, self.sampler = calculate_KDE_sampler(y_train, KDE_width=self.KDE_width, Min_Dens_cap=self.Min_Dens_cap)

            # Save the original training data for comparison with KDE sampling
            self.orignal_y_train = y_train
        
        y_train = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).T[0]
        y_val   = self.target_scaler.transform(y_val.reshape(-1, 1)).T[0]
        y_test  = self.target_scaler.transform(y_test.reshape(-1, 1)).T[0]

        self.pretty_feature_names = prettify_feature_names(p_scaler.get_feature_names_out(), pretty_label_dict, extra_feature_label_dict, self.n_particles)

        if self.using_extra_features: 
            joined_features = np.append(p_scaler.get_feature_names_out(), m_scaler.get_feature_names_out())
            self.pretty_feature_names = prettify_feature_names(joined_features, pretty_label_dict, extra_feature_label_dict, self.n_particles)
    
        self.__plot_distributions(p_train, m_train, left_tail_ind, right_tail_ind, stage="Scaled_Data", scaled=True) # Plot data after scaling

        plot_particle_distributions(left_tail_ind, right_tail_ind, y_train, title='Target_feature', n_bins=40, no_particle_name=True, 
                                    folder_name= self.result_dir + 'Scaled_Data')

        # Compute number of features in the input and interaction tokens
        self.input_dim  = p_train.shape[1] // self.n_particles 
        if self.compute_pairing_tokens:
            self.event_dim  = m_train.shape[2]

        # Creat the tokens with dimensions (num_events, num_features, num_particles)
        p_train = p_train.reshape(p_train.shape[0], self.input_dim, -1)
        p_val   = p_val.reshape(p_val.shape[0], self.input_dim, -1)
        p_test  = p_test.reshape(p_test.shape[0], self.input_dim, -1)

        # Now dimensions are (num_events, num_particles, num_features)
        p_train = p_train.transpose(0, 2, 1)
        p_val = p_val.transpose(0, 2, 1)
        p_test = p_test.transpose(0, 2, 1)

        logger.info(f"Shape of particle input data {p_train.shape}")

        # If extra features are used, they need to be added to expanded into each particle token
        if self.using_extra_features:
            extra_train_expanded = np.expand_dims(m_train, axis=1)  # (batch_size, 1, num_extra_features)
            extra_train_tiled = np.tile(extra_train_expanded, (1, p_train.shape[1], 1)) 
            p_train = np.concatenate([p_train, extra_train_tiled], axis=2)
            
            extra_val_expanded = np.expand_dims(m_val, axis=1)  # (batch_size, 1, num_extra_features)
            extra_val_tiled = np.tile(extra_val_expanded, (1, p_val.shape[1], 1)) 
            p_val = np.concatenate([p_val, extra_val_tiled], axis=2)
            
            extra_test_expanded = np.expand_dims(m_test, axis=1)  # (batch_size, 1, num_extra_features)
            extra_test_tiled = np.tile(extra_test_expanded, (1, p_test.shape[1], 1)) 
            p_test = np.concatenate([p_test, extra_test_tiled], axis=2)

            self.input_dim += m_train.shape[1]
            
            logger.info(f'Shape of particle training tokens post extra features: {p_train.shape}')

        # Again passing in "dummy" m_train when no extra features or interaction tokens are computed
        self.train_dataset = self._make_dataset(p_train, m_train, y_train)
        self.val_dataset   = self._make_dataset(p_val, m_val, y_val)
        self.test_dataset  = self._make_dataset(p_test, m_test, y_test)

        # === Log sizes === #
        logger.info(f"The length of each token is: {self.input_dim}")
        logger.info(f"Training dataset size: {len(y_train)}")
        logger.info(f"Validation dataset size: {len(y_val)}")
        logger.info(f"Test dataset size: {len(y_test)}")


    def train_dataloader(self):
        if self.inverse_sampling:
            data_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, sampler=self.sampler, num_workers=self.train_num_workers, persistent_workers=True, drop_last=True, pin_memory=True )

            weights = self.sampler.weights.detach().cpu().numpy()
            y_vals = []

            if self.compute_pairing_tokens: 
                for i, (_, _, y_batch) in enumerate(data_loader):
                    y_vals.extend(y_batch.cpu().numpy())
        
            else: 
                for i, (_, y_batch) in enumerate(data_loader):
                    y_vals.extend(y_batch.cpu().numpy())
            
            y_vals = self.target_scaler.inverse_transform(np.array(y_vals).reshape(-1,1)).flatten()

            weights = np.minimum(1.0 / (self.density + 1e-8), self.Min_Dens_cap)
            plot_kde_and_inverse_weights(self.orignal_y_train, self.density, weights, save_path=self.result_dir + '/KDE_weights.png', xlim=(60, 120) )
            plot_resampled_distributions(self.orignal_y_train, y_vals, self.result_dir + '/KDE.png')
            
            return data_loader

        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_num_workers, persistent_workers=True, drop_last=True, pin_memory=True)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, persistent_workers=True, drop_last=True, pin_memory=True)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)


    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)


    # Private class functions
    def _make_dataset(self, p, m, y):
        if self.compute_pairing_tokens:
            return TensorDataset(to_float_tensor(p), to_float_tensor(m), to_float_tensor(y))
        return TensorDataset(to_float_tensor(p), to_float_tensor(y))

        
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
                
