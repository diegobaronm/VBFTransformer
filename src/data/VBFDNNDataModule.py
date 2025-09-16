import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import polars as pl
import numpy as np
from loguru import logger
import h5py
from numpy.lib import recfunctions as rfn
from omegaconf import DictConfig

from src.data.DataScaler import DNNScaler

class VBFDNNDataModule(L.LightningDataModule):
    def __init__(self, cfg_object : DictConfig):
        super().__init__()
        # User-defined parameters
        self.signal_path = cfg_object.dataset.signal_path
        self.background_path = cfg_object.dataset.background_path
        self.train_num_workers = cfg_object.dataset.train.num_workers
        self.val_num_workers = cfg_object.dataset.val.num_workers
        self.train_batch_size = cfg_object.dataset.train.batch_size
        self.val_batch_size = cfg_object.dataset.val.batch_size

        # Other parameters
        max_particles = 7
        if cfg_object.model.n_particles > max_particles:
            raise ValueError(f"n_particles must be less than or equal to {max_particles}.")
        self.n_particles = cfg_object.model.n_particles
        self.n_features = self.n_particles * 7 # 7 features per particle
        self.feature_names = []

        
    def setup(self, stage: str):
        logger.info("Setting up the data module...")
        # Load the data using H5
        signal_file = h5py.File(self.signal_path,'r')
        bg_file = h5py.File(self.background_path,'r')

        signal_dataset_inputs = signal_file['INPUTS']['PARTICLES'][:,:self.n_particles]
        signal_inputs = rfn.structured_to_unstructured(signal_dataset_inputs)

        background_dataset_inputs = bg_file['INPUTS']['PARTICLES'][:,:self.n_particles]
        background_inputs = rfn.structured_to_unstructured(background_dataset_inputs)


        def flat_inputs(input_array, n_particles):
            output_array = input_array[:,:n_particles,:]
            output_array = output_array.transpose(0,2,1)
            output_array = output_array.reshape(-1, self.n_features)

            return output_array

        # define the features you are interested in
        df_signal_filtered = flat_inputs(signal_inputs, self.n_particles)
        df_background_filtered = flat_inputs(background_inputs, self.n_particles)
        logger.info(f"Number of features: {self.n_features}")

        # Define the feature names
        keys = list(signal_file['INPUTS']['PARTICLES'].dtype.names)
        self.feature_names = [f"{key}{i}" for key in keys for i in range(self.n_particles)]
        logger.info(f"Feature names: {self.feature_names}")
        
        # Set targets for training
        y_signal     = np.ones(len(df_signal_filtered))
        y_background = np.zeros(len(df_background_filtered))

        # Print number of events
        logger.info(f"Number of signal events: {len(df_signal_filtered)}")
        logger.info(f"Number of background events: {len(df_background_filtered)}")
        logger.info(f"Total number of events: {len(df_signal_filtered) + len(df_background_filtered)}")

        # Combine the dataframes as one big numpy array
        input_data = np.concatenate((df_signal_filtered, df_background_filtered), axis=0)
        target     = np.concatenate((y_signal, y_background), axis=0)

        # split data into train, validation, and test sets (You can also do the shuffle here, if not shuffled before)
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_indices, test_indices = next(sss.split(input_data, target))

        X_train, y_train = input_data[train_indices], target[train_indices]
        X_temp, y_temp = input_data[test_indices], target[test_indices]
        X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True)

        # Scale the data
        scaler = DNNScaler
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        # As tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
        X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)
        y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)
        
        # Fix the nan padding
        X_train_tensor = torch.nan_to_num(X_train_tensor, nan=-2)
        X_val_tensor = torch.nan_to_num(X_val_tensor, nan= -2)
        X_test_tensor = torch.nan_to_num(X_test_tensor, nan=-2)

        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
        self.test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

        # Print dataset sizes
        logger.info(f"Training dataset size: {len(y_train_tensor)}")
        logger.info(f"Validation dataset size: {len(y_val_tensor)}")
        logger.info(f"Test dataset size: {len(y_test_tensor)}")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_num_workers, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, persistent_workers=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=4, persistent_workers=True)
