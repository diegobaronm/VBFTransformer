import lightning as L
import torch

from ML.VBFTransformer.src.models.Model import VBFTransformer

def predict(datamodule):
    # Figure out the device to use
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # Define the trainer
    trainer = L.Trainer(accelerator=device)

    # Predict
    model = VBFTransformer(datamodule.n_features)
    predictions = trainer.predict(model, datamodule=datamodule)

    return predictions