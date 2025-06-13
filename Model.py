import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self, N_input_features): # You can add more parameters here, such that the size of all layers can be
        # defined in the constructor
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SimpleDNN, self).__init__()
        self.linear1 = nn.Linear(N_input_features, N_input_features)
        self.linear2 = nn.Linear(N_input_features, 30)
        self.linear3 = nn.Linear(30, 20)
        self.linear4 = nn.Linear(20, 1)
        self.dropput = nn.Dropout()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # Compute the forward pass.
        # The first layer is self.linear1, then we apply the ReLU activation function
        x1     = F.relu(self.linear1(x))
        x1dp = self.dropput(x1)
        # layer2
        x2     = F.relu(self.linear2(x1dp))
        x2dp = self.dropput(x2)
        #layer 3
        x3     = F.relu(self.linear3(x2dp))
        x3dp   = self.dropput(x3)
        # The final layer is self.linear24 then we apply the sigmoid activation function to get our final output
        y_pred = F.sigmoid(self.linear4(x3dp))
        return y_pred
    
import lightning as L
import torch.optim as optim

class VBFTransformer(L.LightningModule):
    def __init__(self, N_features):
        super().__init__()
        self.model = SimpleDNN(N_features)
        self.loss_fn = nn.BCELoss(reduction='mean')

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.02, weight_decay=0.001)
        return optimizer