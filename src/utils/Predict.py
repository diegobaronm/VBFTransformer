import lightning as L
import torch
import pandas as pd
from omegaconf import DictConfig

from src.utils.utils import set_exection_device, get_latest_checkpoint_path, load_checkpoint_into_model

def predict(datamodule, model_class, cfg: DictConfig):
    # Figure out the device to use
    device = set_exection_device(cfg.general.device)
    # Define the trainer
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)
    ckpt_path = get_latest_checkpoint_path(cfg.performance.model_ckpt_path)

    # Instantiate the model
    model = model_class(cfg)
 
    # Prepare datamodule and model (Needs to be done by hand as regression models are only set up after the datamodule has been setup)
    datamodule.setup(stage="predict")
    model.setup(stage="predict", datamodule=datamodule)

    # Load checkpoint into the instance
    model = load_checkpoint_into_model(model, ckpt_path)
    
    # Move model to the correct device
    model.to(device)
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