import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import polars as pl
import numpy as np
from loguru import logger

class VBFTransformerDataModule(L.LightningDataModule):
    def __init__(self, signal_path: str, backgorund_path: str,
                n_particles: int = 7,
                train_num_workers: int = 4,
                val_num_workers: int = 4,
                train_batch_size: int = 1000,
                val_batch_size: int = 500):
        super().__init__()
        # User-defined parameters
        self.signal_path = signal_path
        self.background_path = backgorund_path
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        # Other parameters
        max_particles = 7
        if n_particles > max_particles:
            raise ValueError(f"n_particles must be less than or equal to {max_particles}.")
        self.n_features = n_particles * 7 # 7 features per particle

    def setup(self, stage: str):
        logger.info("Setting up the data module...")
        # Load the data using Polars
        signal_df = pl.read_csv(self.signal_path)
        background_df = pl.read_csv(self.background_path)

        # define the features you are interested in
        input_features = signal_df.columns
        self.n_features = len(input_features)
        df_signal_filtered = signal_df[input_features]
        df_background_filtered = background_df[input_features]

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

        # Scale the data to lie between -1 and 1 using sklearn StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
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
        return DataLoader(self.test_dataset, batch_size=32*4096, shuffle=False, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32*4096, shuffle=False, num_workers=4, persistent_workers=True)