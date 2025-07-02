import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from ML.VBFTransformer.src.models.Model import VBFTransformer

def train(DM, cfg: DictConfig):
    """
    This function is used to train the model.
    """

    # Figure out the device to use
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # Define the model
    model = VBFTransformer(DM.n_features)
    
    # Define the trainer
    trainer = L.Trainer(max_epochs=cfg.train.n_epochs, accelerator=device)
    
    # Train the model
    trainer.fit(model=model, datamodule=DM)