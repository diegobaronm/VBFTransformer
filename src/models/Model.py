import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import torch.optim as optim
import torch
import pandas as pd
import numpy as np

from src.utils.utils import check_and_overwrite_result_path

class SimpleDNN(nn.Module):
    def __init__(self, N_input_features, dropout_probability : float): # You can add more parameters here, such that the size of all layers can be
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
        self.dropput = nn.Dropout(p=dropout_probability)

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

class VBFTransformer(L.LightningModule):
    def __init__(self, N_features, dropout_probability : float, learning_rate: float):
        super().__init__()
        # Model parameters
        self.learning_rate = learning_rate
        self.model = SimpleDNN(N_features, dropout_probability)
        self.loss_fn = nn.BCELoss(reduction='mean',)
        # Metrics
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="binary", num_classes=2, threshold=0.5)
        self.roc = torchmetrics.ROC(task="binary",thresholds=100)
        self.feature_importance_column_dic = {} # A map between feature names and an integer for the column index
        self.feature_importance = {} # To hold the metric for each feature name
        # We do the filling of the feature importance in the setup method, because we need to know the feature names first.
        # They come from the data module, which is set up before the model.

        # Scores
        self.signal_scores = torchmetrics.CatMetric()
        self.background_scores = torchmetrics.CatMetric()

        # Results
        self.result_dir = 'results/'

    def setup(self, stage): # This is always called after the data module is setup.
        self.feature_names = self.trainer.datamodule.feature_names
        i = 0
        for name in self.feature_names:
            # Initialize a metric for each feature importance
            self.feature_importance[name] = torchmetrics.AUROC(task="binary", thresholds=100)
            self.feature_importance_column_dic[name] = i
            i += 1
        self.feature_importance['nominal'] = torchmetrics.AUROC(task="binary", thresholds=100)

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
        return {"labels" : y, "predictions" : self.model(x)}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        pred = y_hat.squeeze()

        # Accuracy
        self.accuracy.update(pred, y)
        # Confusion Matrix
        self.confusion_matrix.update(pred, y)
        # ROC
        self.roc.update(pred, y.int())
        # AUROC
        self.feature_importance['nominal'].to(self.device)
        self.feature_importance['nominal'].update(pred, y.int())
        # Feature Importance
        for feature_name, column_index in self.feature_importance_column_dic.items():
            self.feature_importance[feature_name].to(self.device)
            scrambled_column = x[:, column_index].clone()  # Clone to avoid modifying the original tensor
            # Shuffle the column to create a scrambled version
            scrambled_column = scrambled_column[torch.randperm(scrambled_column.size(0))]
            # Update the feature importance metric with the scrambled column
            scrambled_x = x.clone()
            scrambled_x[:, column_index] = scrambled_column
            # Calculate the prediction with the scrambled feature
            scrambled_pred = self.model(scrambled_x)
            scrambled_pred = scrambled_pred.squeeze()
            self.feature_importance[feature_name].update(scrambled_pred, y.int())

        

        # Store scores for later use
        class_scores = {"signal": pred[y.int() == 1], "background": pred[y.int() == 0]}
        self.signal_scores.update(class_scores["signal"])
        self.background_scores.update(class_scores["background"])


    def on_test_epoch_end(self):
        # Log the metrics
        # Log accuracy
        self.log('test_accuracy', self.accuracy.compute())

        # Log Confusion Matrix
        confmat = self.confusion_matrix.compute()
        self.log('test_confmat_00', float(confmat[0][0]))
        self.log('test_confmat_01', float(confmat[0][1]))
        self.log('test_confmat_10', float(confmat[1][0]))
        self.log('test_confmat_11', float(confmat[1][1]))

        print('Saving confusion matrix plot...')
        fig_, ax_ = self.confusion_matrix.plot()

        save_path = check_and_overwrite_result_path(self.result_dir+'confusion_matrix.png')
        fig_.savefig(save_path)

        # Log ROC
        print('Saving ROC curve plot...')
        fig_, ax_ = self.roc.plot(score=True)
        save_path = check_and_overwrite_result_path(self.result_dir+'roc_curve.png')
        fig_.savefig(save_path)

        # Print scores plot
        signal_scores = self.signal_scores.compute()
        background_scores = self.background_scores.compute()
        import matplotlib.pyplot as plt
        print('Saving signal and background scores plot...')
        plt.figure(figsize=(10, 5))
        plt.hist(signal_scores.cpu().numpy(), bins=50, alpha=0.5, label='Signal Scores', color='blue', density=True)
        plt.hist(background_scores.cpu().numpy(), bins=50, alpha=0.5, label='Background Scores', color='red', density=True)
        plt.xlabel('Scores')
        plt.ylabel('Number of Events')
        plt.title('Signal and Background Scores')
        plt.legend()
        save_path = check_and_overwrite_result_path(self.result_dir+'signal_background_scores.png')
        plt.savefig(save_path)

        # Log AUROC
        self.feature_importance['nominal'].to(self.device)
        nominal_auc = self.feature_importance['nominal'].compute()
        self.log('test_nominal_auc', nominal_auc)

        # Log feature importance and produce a plot
        importance_dict = {} # AUC, percentage difference from nominal
        for feature_name, metric in self.feature_importance.items():
            metric.to(self.device)
            auc_value = metric.compute()
            percentage_difference = 100 * (nominal_auc - auc_value) / nominal_auc
            importance_dict[feature_name] = percentage_difference.cpu().numpy()  # Convert to numpy for easier handling
            self.log(f'test_feature_importance_{feature_name}', percentage_difference)
        # Order the feature importance dictionary by percentage difference
        sorted_indices = np.argsort(list(importance_dict.values()))
        sorted_features = [list(importance_dict.keys())[i] for i in sorted_indices]
        sorted_importances = [importance_dict[feature] for feature in sorted_features]
        # Plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance - Percentage Difference from Nominal AUC')
        plt.title('Feature Importances')
        save_path = check_and_overwrite_result_path(self.result_dir+'feature_importances.png')
        plt.savefig(save_path)

        # Reset metrics for the next epoch
        self.confusion_matrix.reset()
        self.roc.reset()
        self.signal_scores.reset()
        self.background_scores.reset()
        self.feature_importance['nominal'].reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001, amsgrad=True)
        return optimizer