# %%
import pandas as pd
import numpy as np

from rdkit import Chem

import optuna
from optuna.pruners import SuccessiveHalvingPruner

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer



from torch_geometric.nn import GATv2Conv, TransformerConv
from torch_scatter import scatter_mean

from lion_pytorch import Lion

print("cuda", torch.cuda.is_available())  
print(torch.cuda.get_device_name(0)) 

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")

torch.cuda.empty_cache()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.train import MoleculeModel, MoleculeDataModule, GATv2Model, get_metric, save_trial_to_csv, create_hyperopt_dir
from utils.prepare import FeaturizationParameters, MoleculeDataset, MoleculeData


# %%
molecule_dataset = torch.load("../data/QM_10k.pt")

# %%
num_workers = 8
in_features = molecule_dataset[0].x.shape[1]
edge_attr_dim = molecule_dataset[0].edge_attr.shape[1]
max_epochs = 100
patience = 5

# %% [markdown]
# ### Гиперпараметры

# %%
import optuna

def objective(trial):
    # Гиперпараметры для предобработки
    num_preprocess_layers = trial.suggest_int('num_preprocess_layers', 1, 3)
    preprocess_hidden_features = [trial.suggest_categorical(f'preprocess_layer_{i}_size', [32, 64, 128, 256, 512, 1024]) for i in range(num_preprocess_layers)]
    
    # Гиперпараметры для постобработки
    num_postprocess_layers = trial.suggest_int('num_postprocess_layers', 2, 6)
    postprocess_hidden_features = [trial.suggest_categorical(f'postprocess_layer_{i}_size', [32, 64, 128, 256, 512, 1024]) for i in range(num_postprocess_layers)]
    
    # Другие гиперпараметры
    num_heads = [trial.suggest_int(f'num_heads_{i}', 8, 20, step=2) for i in range(2)]
    dropout_rates = [trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.2, step=0.1) for i in range(num_preprocess_layers + 2 + num_postprocess_layers)]
    use_batch_norm = [trial.suggest_categorical(f'use_batch_norm_{i}', [True, False]) for i in range(num_preprocess_layers + 2 + num_postprocess_layers)]
    learning_rate = 8.5e-4
    weight_decay = 2e-4
    step_size = 50
    gamma = 0.1
    batch_size = 128

    #learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    #weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    #step_size = trial.suggest_int('step_size', 10, 200)
    #gamma = trial.suggest_float('gamma', 0.1, 0.9)
    #batch_size = trial.suggest_int('batch_size', 64, 128, step=64)

    # Создание модели с переменными гиперпараметрами
    base_model = GATv2Model(
        atom_in_features=in_features,
        edge_in_features=edge_attr_dim,
        num_preprocess_layers=num_preprocess_layers,
        preprocess_hidden_features=preprocess_hidden_features,
        num_heads=num_heads,
        dropout_rates=dropout_rates,
        activation_fns=[nn.ReLU for _ in range(len(dropout_rates))],  # Для простоты используем ReLU для всех слоев
        use_batch_norm=use_batch_norm,
        num_postprocess_layers=num_postprocess_layers,
        postprocess_hidden_features=postprocess_hidden_features,
        out_features=1
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

    # Обучение модели
    data_module = MoleculeDataModule(molecule_dataset, batch_size=128, num_workers=num_workers)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator='gpu',
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        auto_scale_batch_size=True,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience, verbose=False, mode='min')]
    )
    trainer.fit(model, data_module)

    # Получение потерь на валидационном наборе
    val_loss = trainer.callback_metrics["val_loss"].item()
    trial_value = torch.sqrt(torch.tensor(val_loss)).item()

    # Сохранение результатов испытания
    save_trial_to_csv(trial, hyperopt_dir, trial_value)

    return trial_value

# Начало оптимизации
hyperopt_dir = create_hyperopt_dir()
print(f"Results will be saved in: {hyperopt_dir}")

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Вывод лучших результатов
print(f'Best trial: {study.best_trial.number}')
print(f'Best value (RMSE): {study.best_trial.value}')
for key, value in study.best_trial.params.items():
    print(f'{key}: {value}')



