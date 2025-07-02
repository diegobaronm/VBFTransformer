import lightning as L
import torch

from ML.VBFTransformer.src.models.Model import VBFTransformer

def testing(datamodule):
    # Figure out the device to use
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # Define the trainer
    trainer = L.Trainer(accelerator=device, )

    # Predict
    model = VBFTransformer.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=199-step=200.ckpt', N_features=datamodule.n_features)
    model.eval()  # Set the model to evaluation mode
    trainer.test(model, datamodule=datamodule)

