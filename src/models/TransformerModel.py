import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from omegaconf import DictConfig

from src.utils.utils import check_and_overwrite_result_path

class BasicTransformer(nn.Module):
    def __init__(self, input_dim, n_head, n_layers, dropout_probability):
        super(BasicTransformer, self).__init__()
        # Input embedding head
        self.input_embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 128),
        )
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, dim_feedforward=512, nhead=n_head,batch_first=True),
            num_layers=n_layers, enable_nested_tensor=False)


        self.output_classifier_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def mean_pooling(self,x):
        """
        Mean pooling implementation
        """
        # x: (batch_size, num_tokens, embed_dim)
        return x.mean(dim=1)

    def attention_pooling(self,x):

        """
        Attention pooling implementation
        """
        # x: (batch_size, num_tokens, embed_dim)
        B = x.size(0)
        pool_query = nn.Parameter(torch.randn(1, 1, 32)).to(device)
        query = pool_query.expand(B, -1, -1)  # (B, 1, embed_dim)

        # Compute attention scores
        attn_scores = torch.matmul(query, x.transpose(1, 2))  # (B, 1, num_tokens)
        attn_weights = torch.softmax(attn_scores, dim=-1)     # (B, 1, num_tokens)

        # Weighted sum
        pooled = torch.matmul(attn_weights, x)  # (B, 1, embed_dim)
        return pooled.squeeze(1)

    def forward(self, x):
        x = self.input_embedder(x)
        x = self.transformer_encoder(x)
        # x = self.attention_pooling(x)
        x = self.mean_pooling(x)
        x = self.output_classifier_head(x)
        return F.sigmoid(x)

class VBFTransformer(L.LightningModule):
    def __init__(self, config_object : DictConfig):
        super().__init__()
        # Model parameters
        self.learning_rate = config_object.train.learning_rate
        self.model = BasicTransformer(input_dim=config_object.model.input_dim,
                                       n_head=config_object.model.n_heads,
                                       n_layers=config_object.model.n_layers,
                                       dropout_probability=config_object.train.dropout_probability)
        self.model = torch.compile(self.model)  # Compile the model for better performance
        self.loss_fn = nn.BCELoss(reduction='none')
        # Raw inverse freq
        w0 = 1.0 / 5594716  # 0.001
        w1 = 1.0 / 161959    # 0.1
        weights = torch.tensor([w0, w1], dtype=torch.float32)
        # Normalize to sum to 2 (number of classes)
        self.weights = weights * 2.0 / weights.sum()
        self.weights.to(self.device)  # Move weights to the device
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
        weight = torch.where(y == 1, self.weights[1], self.weights[0])
        loss = (loss * weight).mean()  # Apply the weights to the loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat.squeeze(), y)
        weight = torch.where(y == 1, self.weights[1], self.weights[0])
        loss = (loss * weight).mean()  # Apply the weights to the loss
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
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01, amsgrad=True)
        return optimizer