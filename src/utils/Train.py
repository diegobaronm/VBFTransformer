import lightning as L
import torch
from omegaconf import DictConfig
from loguru import logger

from src.utils.utils import set_exection_device
from src.models.TransformerModel import VBFTransformer
from src.models.DNNModel import VBFDNN
        

def train(datamodule, model, cfg: DictConfig):
    """
    This function is used to train the model.
    """

    # Figure out the device to use
    device = set_exection_device(cfg.general.device)

    # Define the model
    model = model(config_object=cfg)
    
    # Define the trainer
    trainer = L.Trainer(max_epochs=cfg.train.n_epochs, accelerator=device)
    
    # Train the model
    trainer.fit(model=model, datamodule=datamodule)