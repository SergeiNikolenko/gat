# %%
import optuna
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer


from lion_pytorch import Lion

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    print("cuda", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    print("CUDA is not available.")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision('medium')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.add_skipatom import add_skipatom_features_to_dataset
from utils.utils import save_trial_to_csv, evaluate_model, create_hyperopt_dir, MoleculeDataModule
from utils.train import MoleculeModel

# %%
dataset = torch.load("../data/QM_137k.pt")
dataset = dataset[:10000]
# %% [markdown]
# ### Гиперпараметры

# %%
import optuna

def objective(trial):
    try:


        top_n = trial.suggest_int('top_n', 3, 7)
        min_count = trial.suggest_float('min_count', 2e5, 2e10, log=True)

        dataset_ex = add_skipatom_features_to_dataset(dataset=dataset, 
                                                   model_file="../skipatom/data/mp_2020_10_09.dim30.model",
                                                   min_count=top_n, 
                                                   top_n=min_count, 
                                                   device='cpu', 
                                                   progress_bar=False, 
                                                   scaler=StandardScaler())

        in_features = dataset_ex[0].x.shape[1]
        out_features = 1
        edge_attr_dim = dataset_ex[0].edge_attr.shape[1]

        preprocess_hidden_features = [128] * 9
        postprocess_hidden_features = [128, 128]

        cheb_hidden_features = [1024, 1024]
        cheb_normalization = ['sym', 'sym']
        K = [10, 16]

        dropout_rates = [0.0] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
        activation_fns = [nn.PReLU] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
        use_batch_norm = [True] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))

        batch_size = 128
        learning_rate = 2.2e-5
        weight_decay = 3e-5

        step_size = 80
        gamma = 0.2



        model = MoleculeModel(
            atom_in_features=in_features,
            edge_attr_dim=edge_attr_dim,
            preprocess_hidden_features=preprocess_hidden_features,
            cheb_hidden_features=cheb_hidden_features,
            K=K,
            cheb_normalizations=cheb_normalization,
            dropout_rates=dropout_rates,
            activation_fns=activation_fns,
            use_batch_norm=use_batch_norm,
            postprocess_hidden_features=postprocess_hidden_features,
            out_features=out_features,
            optimizer_class=Lion,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            step_size=step_size,
            gamma=gamma,
            batch_size=batch_size,
            metric='rmse'
        )

        data_module = MoleculeDataModule(dataset_ex, batch_size=batch_size, num_workers=8)

        early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        timer = Timer()
        logger = pl.loggers.TensorBoardLogger('tb_logs', name='hyperopt/skipatom')

        trainer = pl.Trainer(
            max_epochs=100,
            devices=1,
            accelerator='gpu',
            logger=logger,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[early_stop_callback, timer]
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

study.optimize(objective, n_trials=1000)

print(f'Best trial: {study.best_trial.number}')
print(f'Best value (RMSE): {study.best_trial.value}')
for key, value in study.best_trial.params.items():
    print(f'{key}: {value}')



