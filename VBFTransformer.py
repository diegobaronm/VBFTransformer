import h5py
import pandas as pd
import torch
import gc # Garbage collector
import math
import matplotlib.pyplot as plt

import lightning as L

from Train import train
from Predict import predict
from VBFTransformerDataModule import VBFTransformerDataModule

def main(mode):
    
    datamodule = VBFTransformerDataModule('../data/signal_polars.csv','../data/background_polars.csv', n_particles=7)

    if mode == 'train':
        # Train the model
        train(datamodule)
    if mode == 'predict':
       # Predict using the model
       predictions = predict(datamodule)
       print('Number of predictions:', len(predictions))
       for i in predictions:
           print(i)

if __name__ == "__main__":
    # This is the main entry point for the application.
    import argparse
    parser = argparse.ArgumentParser(description="Train a VBFTransformer model.")
    parser.add_argument('--mode', help='Mode to run the script in', choices=['train', 'predict'], default='train')
    args = parser.parse_args()

    # Run
    main(args.mode)