import os
import csv


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, MessagePassing
from torch_scatter import scatter_mean
from torch.utils.data import Subset
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_model(model, data_module):
    test_dl = data_module.test_dataloader()
    model.eval()  
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in test_dl:
            y_hat = model(batch.x, batch.edge_index, batch.edge_attr)
            all_pred.extend(y_hat.cpu().numpy())
            all_true.extend(batch.y.cpu().numpy())

    all_pred, all_true = np.array(all_pred), np.array(all_true)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)

    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test R²: {r2:.4f}')

def create_hyperopt_dir(base_dir='hyperopt_'):
    idx = 1
    while True:
        dir_name = f"{base_dir}{idx}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        idx += 1


def save_trial_to_csv(trial, hyperopt_dir, trial_value):
    csv_path = os.path.join(hyperopt_dir, 'optuna_results.csv')
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.path.getsize(csv_path) == 0:  
            headers = ['Trial'] + ['Value'] + [key for key in trial.params.keys()]
            writer.writerow(headers)
        row = [trial.number] + [trial_value] + list(trial.params.values())
        writer.writerow(row)



def get_metric(metric_name):
    if metric_name == 'mse':
        def mse(y_true, y_pred):
            return F.mse_loss(y_pred, y_true)
        return mse

    elif metric_name == 'rmse':
        def rmse(y_true, y_pred):
            return torch.sqrt(F.mse_loss(y_pred, y_true))
        return rmse

    else:
        raise ValueError(f"Неизвестное имя метрики: {metric_name}")

class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=128, val_split=0.1, test_split=0.2, num_workers=1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        indices = list(range(len(self.dataset)))
        train_val_indices, test_indices = train_test_split(indices, test_size=self.test_split, random_state=42)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=self.val_split / (1 - self.test_split), random_state=42)
        
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



class MoleculeModel(pl.LightningModule):
    def __init__(self, base_model, optimizer_class, learning_rate, weight_decay, step_size, gamma, batch_size, metric='rmse'):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])
        self.base_model = base_model
        self.batch_size = batch_size
        self.metric = get_metric(metric)

        self.train_losses = []
        self.val_losses = []

    def forward(self, x, edge_index, edge_attr):
        return self.base_model(x, edge_index, edge_attr)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        loss = self.metric(batch.y, y_hat)
        self.log('train_loss', loss, batch_size=self.batch_size)
        self.train_losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        val_loss = self.metric(batch.y, y_hat)
        self.log('val_loss', val_loss, batch_size=self.batch_size)
        self.val_losses.append(val_loss.item())

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        preds_np = y_hat.detach().cpu().numpy()
        true_values_np = batch.y.detach().cpu().numpy()

        data = []
        start_idx = 0
        for i, num_atoms in enumerate(batch.ptr[:-1]): 
            end_idx = batch.ptr[i+1].item()
            molecule_preds = preds_np[start_idx:end_idx]
            molecule_true_values = true_values_np[start_idx:end_idx]

            data.append({
                'smiles': batch.smiles[i],
                'predictions': molecule_preds,
                'true_values': molecule_true_values
            })

            start_idx = end_idx
        return data

    def on_test_epoch_end(self, outputs):

        all_data = [item for batch_data in outputs for item in batch_data]
        self.df_results = pd.DataFrame(all_data)

        all_predictions = np.concatenate(self.df_results['predictions'].values)
        all_true_values = np.concatenate(self.df_results['true_values'].values)

        rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
        mse = mean_squared_error(all_true_values, all_predictions)
        r2 = r2_score(all_true_values, all_predictions)
        mae = mean_absolute_error(all_true_values, all_predictions)

        self.log('test_rmse', rmse)
        self.log('test_mse', mse)
        self.log('test_r2', r2)
        self.log('test_mae', mae)

        print(f'Test RMSE: {rmse:.4f}')
        print(f'Test MSE: {mse:.4f}')
        print(f'Test R²: {r2:.4f}')
        print(f'Test MAE: {mae:.4f}')

        return self.df_results


class GATv2Model(nn.Module):
    def __init__(self, atom_in_features, edge_in_features, num_preprocess_layers, preprocess_hidden_features, num_heads, dropout_rates, activation_fns, use_batch_norm, num_postprocess_layers, postprocess_hidden_features, out_features):
        super(GATv2Model, self).__init__()

        # Создаем слои предобработки для атомных свойств
        self.atom_preprocess_layers = nn.ModuleList()
        for i in range(num_preprocess_layers):
            layers = []
            in_features = atom_in_features if i == 0 else preprocess_hidden_features[i-1]
            layers.append(nn.Linear(in_features, preprocess_hidden_features[i]))
            if use_batch_norm[i]:
                layers.append(nn.BatchNorm1d(preprocess_hidden_features[i]))
            layers.append(activation_fns[i]())
            layers.append(nn.Dropout(dropout_rates[i]))
            self.atom_preprocess_layers.append(nn.Sequential(*layers))

        # Создаем слои предобработки для свойств связей
        self.edge_preprocess_layers = nn.ModuleList()
        for i in range(num_preprocess_layers):
            layers = []
            in_features = edge_in_features if i == 0 else preprocess_hidden_features[i-1]
            layers.append(nn.Linear(in_features, preprocess_hidden_features[i]))
            if use_batch_norm[i]:
                layers.append(nn.BatchNorm1d(preprocess_hidden_features[i]))
            layers.append(activation_fns[i]())
            layers.append(nn.Dropout(dropout_rates[i]))
            self.edge_preprocess_layers.append(nn.Sequential(*layers))

        # Создаем слои GATv2
        self.gat_conv_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=preprocess_hidden_features[-1] * (2 if i == 0 else num_heads[i - 1]),
                out_channels=preprocess_hidden_features[-1],
                heads=num_heads[i],
                dropout=dropout_rates[num_preprocess_layers + i],
                concat=True
            ) for i in range(len(num_heads))
        ])

        # Создаем слои постобработки
        self.postprocess_layers = nn.ModuleList()
        for i in range(num_postprocess_layers):
            layers = []
            in_features = preprocess_hidden_features[-1] * num_heads[-1] if i == 0 else postprocess_hidden_features[i-1]
            layers.append(nn.Linear(in_features, postprocess_hidden_features[i]))
            if use_batch_norm[num_preprocess_layers + len(num_heads) + i]:
                layers.append(nn.BatchNorm1d(postprocess_hidden_features[i]))
            layers.append(activation_fns[num_preprocess_layers + len(num_heads) + i]())
            layers.append(nn.Dropout(dropout_rates[num_preprocess_layers + len(num_heads) + i]))
            self.postprocess_layers.append(nn.Sequential(*layers))

        self.final_layer = nn.Linear(postprocess_hidden_features[-1], out_features)

    def forward(self, x, edge_index, edge_attr):
        for layer in self.atom_preprocess_layers:
            x = layer(x)

        for layer in self.edge_preprocess_layers:
            edge_attr = layer(edge_attr)

        # Объединяем атомные свойства и свойства связей
        row, col = edge_index
        agg_edge_features = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))
        x = torch.cat([x, agg_edge_features], dim=1)

        # Применяем слои GATv2
        for conv in self.gat_conv_layers:
            x = conv(x, edge_index)

        # Применяем слои постобработки
        for layer in self.postprocess_layers:
            x = layer(x)

        x = self.final_layer(x).squeeze(-1)
        return x
