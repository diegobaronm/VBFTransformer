import lightning as L
import torch
from omegaconf import DictConfig
from loguru import logger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils.utils import set_exection_device
from src.models.TransformerModel import VBFTransformer
from src.models.DNNModel import VBFDNN
import torch.multiprocessing as mp

def train(datamodule, model, cfg: DictConfig):

    callbacks = []
    device = set_exection_device(cfg.general.device) # Figure out the device to use
    
    # Create early stopping callback with patience from config
    if hasattr(cfg.train, 'early_stopping_patience'):
            callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=getattr(cfg.train, 'early_stopping_patience', 10),
            mode="min",
            verbose=False
        ))
    
    if bool(cfg.train.get('optimize', False)):
        torch.set_float32_matmul_precision("high")  # Enable TF32 for matmuls
        torch.backends.cudnn.benchmark = True       # Auto-tune convolution algorithms

    # Reduced the number of open files that each worker creates, needed to avoid exceeding file open limit when using
    # many workers or opening very large files (about 65k files for Linux systems by default)
    mp.set_sharing_strategy('file_system')
    
    # Define the model
    model = model(config_object=cfg)

    # Without specifying the callback, the model will stil produce checkpoints (.ckpt files) in the lightning_logs dir
    trainer = L.Trainer(
        max_epochs=cfg.train.n_epochs,
        accelerator=device,            # Use GPU
        devices="auto",               # Automatically use all available GPUs
        callbacks=callbacks,
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)
