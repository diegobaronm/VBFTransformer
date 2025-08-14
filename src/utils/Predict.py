import lightning as L
import torch
import pandas as pd
from omegaconf import DictConfig

from src.utils.utils import set_exection_device, get_latest_checkpoint_path
from src.models.TransformerRegression import VBFTransformerRegression
from collections import OrderedDict

def predict(datamodule, model, cfg: DictConfig):
    # Figure out the device to use
    device = set_exection_device(cfg.general.device)
    # Define the trainera
    trainer = L.Trainer(accelerator=device, enable_checkpointing=False, logger=False)

    # Load model weights from .pth file instead of checkpoint
    pth_path = "results/Test_mzmmc_DY/weights.pth"  # Update config to use pth_path instead of ckpt_path

    model = VBFTransformerRegression(cfg)  # Assuming your model class takes config_object as parameter
    model.trainer = trainer
    trainer.datamodule = datamodule
    
    # Call setup manually for prediction stage
    model.setup(stage='predict') 

    print("Model keys:")
    for k in model.state_dict().keys():
        print(k)
    
    # Load the state dict from .pth file
    state_dict = torch.load(pth_path, map_location=device)
    print("Checkpoint keys:")
    for k in state_dict.keys():
        print(k)
    
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove 'module.' prefix from keys
        new_key = k.replace("module.", "", 1)
        new_state_dict[new_key] = v

    print(model)
    print(datamodule)
    model.load_state_dict(new_state_dict, strict=True)
    
    # Move model to the correct device
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    predictions = trainer.predict(model, datamodule=datamodule)
    
    # Save predictions to a CSV file
    save_predictions = []
    save_labels = []
    for element in predictions:
        # element["predictions"] and element["labels"] are now numpy arrays
        predictions_array = element["predictions"]
        labels_array = element["labels"]
        
        # Convert numpy arrays to lists
        if predictions_array.ndim == 2:
            # If 2D, flatten to 1D first
            save_predictions += predictions_array.flatten().tolist()
        else:
            save_predictions += predictions_array.tolist()
            
        if labels_array.ndim == 2:
            # If 2D, flatten to 1D first  
            save_labels += labels_array.flatten().tolist()
        else:
            save_labels += labels_array.tolist()
    
    df = pd.DataFrame({
        'predictions': save_predictions,
        'labels': save_labels    
    })
    df.to_csv(model.result_dir + cfg.predict.output_file, index=False)