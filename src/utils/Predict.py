import lightning as L
import torch
import pandas as pd
from omegaconf import DictConfig

from src.utils.utils import set_exection_device, get_latest_checkpoint_path

def predict(datamodule, model, cfg: DictConfig):
    # Figure out the device to use
    device = set_exection_device(cfg.general.device)
    # Define the trainer
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)

    # Load desired model and predict
    ckpt_path = get_latest_checkpoint_path(cfg.predict.model_ckpt_path)
    model = model.load_from_checkpoint(ckpt_path, config_object = cfg)
    model.eval()  # Set the model to evaluation mode
    predictions = trainer.predict(model, datamodule=datamodule)

    # Save predictions to a CSV file
    save_predictions = []
    save_labels = []
    for element in predictions:
        save_predictions += element["predictions"].cpu().numpy().tolist()  # Convert tensor to numpy and then to list
        save_labels += element["labels"].cpu().numpy().tolist()  # Convert tensor to numpy and then to list

    df = pd.DataFrame({
        'predictions': save_predictions,
        'labels': save_labels    
    })
    df.to_csv(model.result_dir+cfg.predict.output_file, index=False)