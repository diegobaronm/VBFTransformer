import lightning as L
import torch
from omegaconf import DictConfig

from src.utils.utils import set_exection_device, get_latest_checkpoint_path

def testing(datamodule, model ,cfg: DictConfig):
    # Figure out the device to use
    device = set_exection_device(cfg.general.device)
    # Define the trainer
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)
    ckpt_path = get_latest_checkpoint_path(cfg.performance.model_ckpt_path)

    # Do the evaluation
    model = model.load_from_checkpoint(ckpt_path, config_object = cfg)
    model.eval()  # Set the model to evaluation mode
    trainer.test(model, datamodule=datamodule)

