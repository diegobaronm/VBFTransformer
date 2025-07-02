import lightning as L
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
import polars as pl
import numpy as np

class VBFTransformerDataModule(L.LightningDataModule):
    def __init__(self, signal_path: str, backgorund_path: str, n_particles: int = 7):
        super().__init__()
        self.signal_path = signal_path
        self.background_path = backgorund_path
        max_particles = 7
        if n_particles > max_particles:
            raise ValueError(f"n_particles must be less than or equal to {max_particles}.")
        self.n_features = n_particles * 7 # 7 features per particle

    def setup(self, stage: str):
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


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=160*4096, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32*4096, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32*4096, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32*4096, shuffle=False, num_workers=4)