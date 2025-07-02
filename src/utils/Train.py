import lightning as L
import torch
from omegaconf import DictConfig
from loguru import logger

from src.models.Model import VBFTransformer

def set_exection_device(cfg_device : str):
    if cfg_device == None:
        logger.warning("No device specified in the configuration, defaulting to CPU.")
        return "cpu"

    is_cuda_available = torch.accelerator.is_available()
    if is_cuda_available and cfg_device.lower() == "cuda":
        logger.info("CUDA is available and the configuration specifies CUDA. Using CUDA.")
        return "cuda"
    elif is_cuda_available and cfg_device.lower() == "cpu":
        logger.info("CUDA is available, but the configuration specifies CPU. Using CPU.")
        return "cpu"
    elif not is_cuda_available and cfg_device.lower() == "cuda":
        logger.warning("CUDA is not available, but the configuration specifies CUDA. Using CPU instead.")
        return "cpu"
    elif not is_cuda_available and cfg_device.lower() == "cpu":
        logger.info("CUDA is not available, using CPU as specified in the configuration.")
        return "cpu"
    else:
        logger.error(f"Invalid device specified in the configuration: {cfg_device}. Select between cuda/cpu.")
        raise ValueError()

        

def train(DM, cfg: DictConfig):
    """
    This function is used to train the model.
    """

    # Figure out the device to use
    device = set_exection_device(cfg.general.device)

    # Define the model
    model = VBFTransformer(DM.n_features)
    
    # Define the trainer
    trainer = L.Trainer(max_epochs=cfg.train.n_epochs, accelerator=device)
    
    # Train the model
    trainer.fit(model=model, datamodule=DM)