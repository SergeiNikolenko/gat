# %%
import optuna
from optuna.pruners import SuccessiveHalvingPruner

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from lion_pytorch import Lion

print("cuda", torch.cuda.is_available())  
print(torch.cuda.get_device_name(0)) 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")

torch.cuda.empty_cache()

from utils.train import MoleculeModel, MoleculeDataModule, save_trial_to_csv, create_hyperopt_dir
from utils.prepare import FeaturizationParameters, MoleculeDataset, MoleculeData

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# %%
import os
import csv


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, TransformerConv, ChebConv
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
    def __init__(self, atom_in_features, preprocess_hidden_features, K, dropout_rates, activation_fns, use_batch_norm, postprocess_hidden_features, out_features, optimizer_class, learning_rate, weight_decay, step_size, gamma, batch_size, metric='rmse'):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.metric = self.get_metric(metric)

        self.train_losses = []
        self.val_losses = []

        # Preprocessing layers for atom features
        self.atom_preprocess = nn.ModuleList()
        for i in range(len(preprocess_hidden_features)):
            preprocess_layer = nn.Sequential()
            in_features = atom_in_features if i == 0 else preprocess_hidden_features[i-1]
            preprocess_layer.add_module(f'atom_linear_{i}', nn.Linear(in_features, preprocess_hidden_features[i]))
            if use_batch_norm[i]:
                preprocess_layer.add_module(f'atom_bn_{i}', nn.BatchNorm1d(preprocess_hidden_features[i]))
            preprocess_layer.add_module(f'atom_activation_{i}', activation_fns[i]())
            preprocess_layer.add_module(f'atom_dropout_{i}', nn.Dropout(dropout_rates[i]))
            self.atom_preprocess.append(preprocess_layer)

        self.cheb_convolutions = nn.ModuleList()
        # Первый ChebConv соединяет последний слой предобработки с первым слоем Чебышева
        self.cheb_convolutions.append(ChebConv(
            in_channels=preprocess_hidden_features[-1],
            out_channels=postprocess_hidden_features[0],  # Подключаем к первому слою постобработки
            K=K[0]
        ))
        # Второй ChebConv (если K содержит более одного элемента)
        if len(K) > 1:
            for i in range(1, len(K)):
                self.cheb_convolutions.append(ChebConv(
                    in_channels=postprocess_hidden_features[i - 1],
                    out_channels=postprocess_hidden_features[i] if i < len(postprocess_hidden_features) else out_features,
                    K=K[i]
                ))

        # Postprocessing layers
        self.postprocess = nn.ModuleList()
        for i in range(len(postprocess_hidden_features)):
            post_layer = nn.Sequential()
            in_features = preprocess_hidden_features[-1] if i == 0 else postprocess_hidden_features[i-1]
            post_layer.add_module(f'post_linear_{i}', nn.Linear(in_features, postprocess_hidden_features[i]))
            if use_batch_norm[len(preprocess_hidden_features) + i]:
                post_layer.add_module(f'post_bn_{i}', nn.BatchNorm1d(postprocess_hidden_features[i]))
            post_layer.add_module(f'post_activation_{i}', activation_fns[len(preprocess_hidden_features) + i]())
            post_layer.add_module(f'post_dropout_{i}', nn.Dropout(dropout_rates[len(preprocess_hidden_features) + i]))
            self.postprocess.append(post_layer)

        self.output_layer = nn.Linear(postprocess_hidden_features[-1], out_features)

    def forward(self, x, edge_index):
        for layer in self.atom_preprocess:
            x = layer(x)

        for conv in self.cheb_convolutions:
            x = F.relu(conv(x, edge_index))

        for layer in self.postprocess:
            x = layer(x)

        x = self.output_layer(x).squeeze(-1)
        return x
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index)
        loss = self.metric(batch.y, y_hat)
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        self.train_losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index)
        val_loss = self.metric(batch.y, y_hat)
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        self.val_losses.append(val_loss.item())

    def test_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index)
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


        print(f'Test RMSE: {rmse:.4f}')
        print(f'Test MSE: {mse:.4f}')
        print(f'Test R²: {r2:.4f}')
        print(f'Test MAE: {mae:.4f}')

        return self.df_results
    

    def get_metric(self, metric_name):
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

# %%
molecule_dataset = torch.load("../data/QM_137k.pt")

# %%
num_workers = 8
in_features = molecule_dataset[0].x.shape[1]
edge_attr_dim = molecule_dataset[0].edge_attr.shape[1]
max_epochs = 100
patience = 2

# %% [markdown]
# ### Гиперпараметры

# %%
import optuna

def objective(trial):

    try:

        in_features = molecule_dataset[0].x.shape[1]
        out_features = 1
        edge_attr_dim = molecule_dataset[0].edge_attr.shape[1]

        preprocess_hidden_features = [128, 128, 128, 128, 128, 128, 128, 128, 128]
        postprocess_hidden_features = [128, 128]
        

        K = [10, 10, trial.suggest_int('K2', 10, 40)]


        optimizer_class = Lion
        learning_rate = 2.2e-5
        weight_decay = 3e-5
        step_size = 80
        gamma = 0.2
        max_epochs = 100
        patience = 5
        batch_size = 128

        dropout_rates = [0.0] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
        activation_fns = [nn.PReLU] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
        use_batch_norm = [True] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))

        model = MoleculeModel(
            atom_in_features=in_features,
            preprocess_hidden_features=preprocess_hidden_features,
            K=K,
            dropout_rates=dropout_rates,
            activation_fns=activation_fns,
            use_batch_norm=use_batch_norm,
            postprocess_hidden_features=postprocess_hidden_features,
            out_features=out_features,
            optimizer_class=optimizer_class,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            step_size=step_size,
            gamma=gamma,
            batch_size=batch_size,
            metric='rmse'
        )

        # Обучение модели
        data_module = MoleculeDataModule(molecule_dataset, batch_size=128, num_workers=num_workers)
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, mode="min")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=1,
            accelerator='gpu',
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            
            callbacks=[early_stop_callback]
        )
        trainer.fit(model, data_module)

        val_loss = trainer.callback_metrics["val_loss"].item()

        save_trial_to_csv(trial, hyperopt_dir, val_loss)

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("CUDA out of memory. Skipping this trial.")
            return float('inf')
        raise  

    return val_loss

#torch.set_float32_matmul_precision('medium')

hyperopt_dir = create_hyperopt_dir()
print(f"Results will be saved in: {hyperopt_dir}")

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())

study.optimize(objective, n_trials=100)

print(f'Best trial: {study.best_trial.number}')
print(f'Best value (RMSE): {study.best_trial.value}')
for key, value in study.best_trial.params.items():
    print(f'{key}: {value}')



