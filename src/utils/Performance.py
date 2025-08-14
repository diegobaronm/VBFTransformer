import lightning as L
import torch
from omegaconf import DictConfig

from src.utils.utils import set_exection_device, get_latest_checkpoint_path

def testing(datamodule, model ,cfg: DictConfig):
    device = set_exection_device(cfg.general.device)
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)
    ckpt_path = get_latest_checkpoint_path(cfg.performance.model_ckpt_path)

    model = model.load_from_checkpoint(ckpt_path, config_object = cfg)
    model.eval()

    results = trainer.test(model, datamodule=datamodule)
    # results is a list of dicts, get the metric you want
    val_loss = results[0].get("val_loss", None)
    if val_loss is None:
        raise ValueError("Validation loss not found in test results")
    return val_loss

