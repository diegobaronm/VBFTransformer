import lightning as L
import torch
from omegaconf import DictConfig
from loguru import logger

from src.models.Model import VBFTransformer
from src.utils.utils import set_exection_device
        

def train(DM, cfg: DictConfig):
    """
    This function is used to train the model.
    """

    # Figure out the device to use
    device = set_exection_device(cfg.general.device)

    # Define the model
    model = VBFTransformer(
        DM.n_features,
        dropout_probability=cfg.train.dropout_probability,
        learning_rate=cfg.train.learning_rate)
    
    # Define the trainer
    trainer = L.Trainer(max_epochs=cfg.train.n_epochs, accelerator=device)
    
    # Train the model
    trainer.fit(model=model, datamodule=DM)