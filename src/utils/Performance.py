import lightning as L
import torch
from omegaconf import DictConfig
from loguru import logger


from src.models.Model import VBFTransformer
from src.utils.Train import set_exection_device

# Get latest checkpoint path
def get_latest_checkpoint_path(checkpoint_dir):
    import os

    # Check if is a directory
    if not os.path.isdir(checkpoint_dir):
        logger.info("Provided a direct file path instead of a directory.")
        # Check if the file exists, if not error out
        if not os.path.isfile(checkpoint_dir):
            logger.error(f"The provided path {checkpoint_dir} is not a file or directory.")
            raise FileNotFoundError()
        else:
            logger.info(f"Using the provided file {checkpoint_dir} as the checkpoint.")
            return checkpoint_dir

    # If it is a directory, find the checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if len(checkpoints) != 1:
        logger.error(f"Expected exactly one checkpoint file in the directory, found {len(checkpoints)}.")
        raise ValueError()
    
    return os.path.join(checkpoint_dir, checkpoints[0])

def testing(datamodule, cfg: DictConfig):
    # Figure out the device to use
    device = set_exection_device(cfg.general.device)
    # Define the trainer
    trainer = L.Trainer(accelerator=device, )
    ckpt_path = get_latest_checkpoint_path(cfg.performance.model_ckpt_path)

    # Predict
    model = VBFTransformer.load_from_checkpoint(ckpt_path, N_features=datamodule.n_features)
    model.eval()  # Set the model to evaluation mode
    trainer.test(model, datamodule=datamodule)

