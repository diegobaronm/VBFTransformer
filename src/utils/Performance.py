import lightning as L
import torch
from omegaconf import DictConfig

from src.utils.utils import set_exection_device, get_latest_checkpoint_path, load_checkpoint_into_model

def testing(datamodule, model_class, cfg: DictConfig):
    # Figure out the device to use
    device = set_exection_device(cfg.general.device)
    # Define the trainer
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)
    ckpt_path = get_latest_checkpoint_path(cfg.performance.model_ckpt_path)

    # Instantiate the model
    model = model_class(cfg)

    # Prepare datamodule and model (Needs to be done by hand as regression models are only set up after the datamodule has been setup)
    datamodule.setup(stage="performance")
    model.setup(stage="performance", datamodule=datamodule)

    # Load checkpoint into the instance
    model = load_checkpoint_into_model(model, ckpt_path)
    model.eval()
    trainer.test(model, datamodule=datamodule)
