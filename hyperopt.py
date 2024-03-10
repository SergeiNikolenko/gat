# %%
import os
import pandas as pd
import numpy as np
import csv

from rdkit import Chem

import optuna
from optuna.pruners import SuccessiveHalvingPruner

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from torch_geometric.nn import GATv2Conv
from lion_pytorch import Lion

print("cuda", torch.cuda.is_available())  
print(torch.cuda.get_device_name(0)) 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")

torch.cuda.empty_cache()

from utils.train import MoleculeModel, MoleculeDataModule
from utils.prepare import load_dataset


# %%
class GATv2Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, dropout_rate, activation_fn):
        super(GATv2Model, self).__init__()
        self.conv1 = GATv2Conv(in_channels=in_features, out_channels=hidden_features, heads=num_heads, dropout=dropout_rate, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_features * num_heads)
        self.prelu = nn.PReLU()
        self.conv2 = GATv2Conv(in_channels=hidden_features * num_heads, out_channels=out_features, heads=1, concat=False)
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.prelu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index).squeeze()
        return x


# %%
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


# %%
molecule_dataset = load_dataset("../data/QM_10k.pt")

# %%
num_workers = 8
in_features = molecule_dataset[0].x.shape[1]
max_epochs = 250
patience = 25

# %%
def objective(trial):
    hidden_features = trial.suggest_int('hidden_features', 32, 512, log=True)
    num_heads = trial.suggest_int('num_heads', 1, 12)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    step_size = trial.suggest_int('step_size', 10, 100)
    gamma = trial.suggest_float('gamma', 0.1, 0.9)
    batch_size = trial.suggest_int('batch_size', 32, 512, step=16)


    activation_name = trial.suggest_categorical('activation_fn', ['relu'])
    activation_fn = getattr(F, activation_name)

    data_module = MoleculeDataModule(molecule_dataset, batch_size=batch_size, num_workers=num_workers)

    base_model = GATv2Model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=1,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        activation_fn=activation_fn  
    )

    model = MoleculeModel(
        base_model=base_model,
        optimizer_class=Lion,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        batch_size=batch_size,
        metric='rmse'  
    )

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=patience, verbose=False, mode='min')

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,  
        accelerator='gpu',
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[early_stop_callback]
    )


    trainer.fit(model, data_module)

    val_loss = trainer.callback_metrics["val_loss"].item()
    trial_value = torch.sqrt(torch.tensor(val_loss))

    save_trial_to_csv(trial, hyperopt_dir, trial_value.item())

    return trial_value.item()

hyperopt_dir = create_hyperopt_dir()
print(f"Results will be saved in: {hyperopt_dir}")

pruner = SuccessiveHalvingPruner()

study = optuna.create_study(direction='minimize', pruner=pruner)
study.optimize(objective, n_trials=1000)

print(f'Best trial: {study.best_trial.number}')
print(f'Best value (RMSE): {study.best_trial.value}')
for key, value in study.best_trial.params.items():
    print(f'{key}: {value}')



