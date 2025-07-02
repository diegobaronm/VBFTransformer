import lightning as L
import torch
from omegaconf import DictConfig


from src.models.Model import VBFTransformer
from src.utils.utils import set_exection_device, get_latest_checkpoint_path

def testing(datamodule, cfg: DictConfig):
    # Figure out the device to use
    device = set_exection_device(cfg.general.device)
    # Define the trainer
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)
    ckpt_path = get_latest_checkpoint_path(cfg.performance.model_ckpt_path)

    # Predict
    model = VBFTransformer.load_from_checkpoint(ckpt_path,
                                                N_features=datamodule.n_features,
                                                dropout_probability=cfg.train.dropout_probability,
                                                learning_rate=cfg.train.learning_rate)
    model.eval()  # Set the model to evaluation mode
    trainer.test(model, datamodule=datamodule)

