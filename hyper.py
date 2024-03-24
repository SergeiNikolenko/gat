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

from utils.train import MoleculeModel, MoleculeDataModule, GATv2Model, get_metric, save_trial_to_csv, create_hyperopt_dir
from utils.prepare import FeaturizationParameters, MoleculeDataset, MoleculeData

import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


# %%
molecule_dataset = torch.load("../data/QM_137k_skip200.pt")

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

        num_preprocess_layers = trial.suggest_int('num_preprocess_layers', 9, 9)
        num_postprocess_layers = trial.suggest_int('num_postprocess_layers', 2, 2)
        
        preprocess_size = trial.suggest_categorical('preprocess_size', [64, 128, 256])
        postprocess_size = trial.suggest_categorical('postprocess_size', [64])
        
        preprocess_hidden_features = [preprocess_size] * num_preprocess_layers
        postprocess_hidden_features = [postprocess_size] * num_postprocess_layers

        num_heads = [trial.suggest_int(f'num_heads_{i}', 16, 32, step=2) for i in range(2)]

        dropout_rates = [trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.1) for i in range(num_preprocess_layers + 2 + num_postprocess_layers)]
        use_batch_norm = [trial.suggest_categorical(f'use_batch_norm_{i}', [True]) for i in range(num_preprocess_layers + 2 + num_postprocess_layers)]
        learning_rate = 2.2e-5
        weight_decay = 3e-5
        step_size = 80
        gamma = 0.2
        batch_size = 64

        #learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        #weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        #step_size = trial.suggest_int('step_size', 10, 200, step=10)
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
            activation_fns=[nn.ReLU for _ in range(len(dropout_rates))],  # ReLU для всех слоев
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

torch.set_float32_matmul_precision('medium')

hyperopt_dir = create_hyperopt_dir()
print(f"Results will be saved in: {hyperopt_dir}")

study = optuna.create_study(direction='minimize', pruner=optuna.pruners.SuccessiveHalvingPruner())

study.optimize(objective, n_trials=10000,)

print(f'Best trial: {study.best_trial.number}')
print(f'Best value (RMSE): {study.best_trial.value}')
for key, value in study.best_trial.params.items():
    print(f'{key}: {value}')


