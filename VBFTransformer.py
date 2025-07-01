import h5py
import pandas as pd
import torch
import gc # Garbage collector
import math
import matplotlib.pyplot as plt
import torchmetrics

import lightning as L

# Hydra - for CLI configuration management
import hydra
from omegaconf import DictConfig, OmegaConf

# Import local modules
from Train import train
from Predict import predict
from Performance import testing
from VBFTransformerDataModule import VBFTransformerDataModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    datamodule = VBFTransformerDataModule(cfg.dataset.signal_path, cfg.dataset.background_path, n_particles=7)

    if cfg.opts.mode == 'train':
        # Train the model
        train(datamodule)

    if cfg.opts.mode == 'predict':
       # Predict using the model
       predictions = predict(datamodule)

       # Save predictions to a CSV file
       save_predictions = []
       save_labels = []
       for element in predictions:
           save_predictions += element["predictions"].cpu().numpy().tolist()  # Convert tensor to numpy and then to list
           save_labels += element["labels"].cpu().numpy().tolist()  # Convert tensor to numpy and then to list

       df = pd.DataFrame({
           'predictions': save_predictions,
           'labels': save_labels    
       })
       df.to_csv('predictions.csv', index=False)

    if cfg.opts.mode == 'performance':
        testing(datamodule)

if __name__ == "__main__":
    # Run
    main()