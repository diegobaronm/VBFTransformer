# === Core Libraries === #
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from loguru import logger
import lightning as L

# === Data Processing === #
from src.data.DataScaler import create_custom_scaler, scaler_map, get_scalers_from_config
from src.data.DataHelpers import calculate_KDE_sampler, load_and_filter_data, prepare_input_data, get_full_feature_index_ranges, to_float_tensor
from src.utils.Plotting import plot_particle_distributions, plot_kde_and_inverse_weights, plot_resampled_distributions
from src.utils.PrettyPrinting import prettify_feature_names
from src.data.DataFormat import MetadataIndex, particle_feature_dict, extra_feature_dict, pretty_label_dict, extra_feature_label_dict


class VBFDNNRegressionDataModule(L.LightningDataModule):
    def __init__(self, cfg_object: DictConfig):
        super().__init__()

        # === Constants === #
        self.MAX_PARTICLES = 10
        self.PEAK_RANGE = [91.8 - 5, 91.8 + 5] # +- 5 GeV from Z-peak, Splits data in 3 regions when plotting
 
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
    
        self.num_quantiles = train_cfg.num_quantiles
        
        self.inverse_sampling = dataset_cfg.get('inverse_sampling', False)
        if self.inverse_sampling:
            self.KDE_width = dataset_cfg.KDE_width
            self.Min_Dens_cap = dataset_cfg.Min_Dens_cap

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
        logger.info(f"Target scaler: {self.target_scaler }")

        # Private variable to ensure set up is not done twice when calling training and test scripts back to back
        self._has_setup = False
    
    def setup(self, stage: str):
        if self._has_setup:
            return
        self._has_setup = True

        if self.n_particles > self.MAX_PARTICLES: raise ValueError(f"n_particles must be ≤ {MAX_PARTICLES}")
        logger.info("Setting up the data module...")
    
        # === Load and prepare data === #
        inputs, metadata = load_and_filter_data(self.input_files)
        input_data = prepare_input_data(inputs, metadata, self.n_particles, self.features, self.extra_features)
        target = metadata[:, MetadataIndex.MZ_TRUTH.value]  # Monte Carlo truth mass
    
        logger.info(f"Particle Features inputted: {self.features}")
    
        # === Split data (80 % Train, 10% val, 10% test) === # 
        indices = np.arange(len(input_data))
        X_train, X_temp, y_train, y_temp, _, idx_temp = train_test_split( input_data, target, indices, test_size=0.2, random_state=0, shuffle=True )
        X_val, X_test, y_val, y_test, _, idx_test = train_test_split( X_temp, y_temp, idx_temp, test_size=0.5, shuffle=True, random_state=0 )
        
        # === Peak filtering and quantiles === #
        left_tail_ind = (y_train < self.PEAK_RANGE[0])
        right_tail_ind = (y_train > self.PEAK_RANGE[1])
        self.quantiles = np.quantile(y_train, np.linspace(0, 1, self.num_quantiles + 1))
        
        self.__plot_distributions(left_tail_ind, right_tail_ind, X_train, stage="Raw_Data", scaled=False) # Plot data before scaling

        plot_particle_distributions(left_tail_ind, right_tail_ind, y_train, title='Target_feature', n_bins=40, no_particle_name=True,folder_name= self.result_dir + 'Raw_Data')

        all_feature_indices = get_full_feature_index_ranges(self.particle_feature_names, self.extra_feature_names, self.n_particles)
        scaler = create_custom_scaler(all_feature_indices, self.scaling_dict, self.extra_scaling_dict)
    
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

         # Now do the inverse scaling and change the distribution of y_train: 
        if self.inverse_sampling: 
            logger.info(f'Creating a KDE sampler')
            self.density, self.sampler = calculate_KDE_sampler(y_train, KDE_width=self.KDE_width, Min_Dens_cap=self.Min_Dens_cap)

            # Save the original training data for comparison with KDE sampling
            self.orignal_y_train = y_train
        
        y_train = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).T[0]
        y_val   = self.target_scaler.transform(y_val.reshape(-1, 1)).T[0]
        y_test  = self.target_scaler.transform(y_test.reshape(-1, 1)).T[0]

        plot_particle_distributions(left_tail_ind, right_tail_ind, y_train, title='Target_feature', n_bins=40, no_particle_name=True, folder_name= self.result_dir + 'Scaled_Data')
    
        self.pretty_feature_names = prettify_feature_names(scaler.get_feature_names_out(), pretty_label_dict, extra_feature_label_dict, self.n_particles)
    
        self.__plot_distributions(left_tail_ind, right_tail_ind, X_train, stage="Scaled_Data", scaled=True) # Plot data after scaling
    
        # === Convert to tensors and dataset objects === #
        self.train_dataset = TensorDataset(to_float_tensor(X_train), to_float_tensor(y_train))
        self.val_dataset   = TensorDataset(to_float_tensor(X_val),   to_float_tensor(y_val))
        self.test_dataset  = TensorDataset(to_float_tensor(X_test),  to_float_tensor(y_test))
    
        # === Log sizes === #
        logger.info(f"Training dataset size: {len(y_train)}")
        logger.info(f"Validation dataset size: {len(y_val)}")
        logger.info(f"Test dataset size: {len(y_test)}")

    def train_dataloader(self):
        if self.inverse_sampling:
            data_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, sampler=self.sampler, num_workers=self.train_num_workers, persistent_workers=True, drop_last=True, pin_memory=True )

            weights = self.sampler.weights.detach().cpu().numpy()
            y_vals = []

            for i, (_, y_batch) in enumerate(data_loader):
                y_vals.extend(y_batch.cpu().numpy())
            
            y_vals = self.target_scaler.inverse_transform(np.array(y_vals).reshape(-1,1)).flatten()
            weights = np.minimum(1.0 / (self.density + 1e-8), self.Min_Dens_cap)
            plot_kde_and_inverse_weights(self.orignal_y_train, self.density, weights, save_path=self.result_dir + '/KDE_weights.png', xlim=(60, 120) )
            
            plot_resampled_distributions(self.orignal_y_train, y_vals, self.result_dir + '/KDE.png')
            
            return data_loader

        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_num_workers, persistent_workers=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, persistent_workers=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    # Private Class functions
    def __plot_distributions(self, left_tail_ind, right_tail_ind, X, stage="Raw_Data", scaled=False):
        folder = self.result_dir + stage
        num_extra = len(self.extra_feature_names)
    
        if not scaled:
            # Plot particle features
            for idx, name in enumerate(self.features):
                start = idx * self.n_particles
                end = start + self.n_particles
                plot_particle_distributions(
                    left_tail_ind, right_tail_ind,
                    X[:, start:end],
                    title=self.particle_feature_names[idx],
                    n_bins=40,
                    folder_name=folder
                )
    
            # Plot extra features
            last_index = len(self.features) * self.n_particles
            for idx, name in enumerate(self.extra_feature_names):
                plot_particle_distributions(
                    left_tail_ind, right_tail_ind,
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
                    left_tail_ind, right_tail_ind,
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
                    left_tail_ind, right_tail_ind,
                    data_extra,
                    title=self.pretty_feature_names[i],
                    n_bins=40,
                    no_particle_name=True,
                    folder_name=folder
                )
    
    
    
    
        
        