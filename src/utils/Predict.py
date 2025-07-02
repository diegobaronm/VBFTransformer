import lightning as L
import torch
import pandas as pd

from ..models.Model import VBFTransformer

def predict(datamodule):
    # Figure out the device to use
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # Define the trainer
    trainer = L.Trainer(accelerator=device)

    # Predict
    model = VBFTransformer(datamodule.n_features)
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
    df.to_csv(model.result_dir+'predictions.csv', index=False)