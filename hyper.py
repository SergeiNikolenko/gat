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

# %%
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, TransformerConv, ChebConv
from torch_scatter import scatter_mean

import torch.nn.functional as F
import pytorch_lightning as pl

from utils.train import MoleculeModel



class AtomEdgeInteraction(nn.Module):
    def __init__(self, in_features, edge_features, out_features, edge_importance=1.0):
        super(AtomEdgeInteraction, self).__init__()
        self.edge_importance = edge_importance
        self.interaction = nn.Linear(in_features + edge_features, out_features)

    def forward(self, x, edge_index, edge_attr):
        # Получение атрибутов связей для каждой связи в графе
        row, col = edge_index
        edge_features = edge_attr * self.edge_importance

        # Комбинирование атрибутов атомов с атрибутами связей
        atom_features = x[row]  # Атрибуты исходящих атомов
        combined_features = torch.cat([atom_features, edge_features], dim=-1)

        # Применение слоя для комбинированных атрибутов
        updated_features = self.interaction(combined_features)

        # Обновление атрибутов атомов
        x = scatter_mean(updated_features, col, dim=0, dim_size=x.size(0))
        return x


class Model(nn.Module):
    def __init__(self, atom_in_features, edge_attr_dim, preprocess_hidden_features, cheb_hidden_features, K, cheb_normalizations, dropout_rates, activation_fns, use_batch_norm, postprocess_hidden_features, out_features):
        super(Model, self).__init__()

        self.atom_preprocess = nn.ModuleList([AtomEdgeInteraction(atom_in_features, edge_attr_dim, preprocess_hidden_features[0])])
        for i in range(1, len(preprocess_hidden_features)):
            layer = nn.Sequential(
                nn.Linear(preprocess_hidden_features[i-1], preprocess_hidden_features[i]),
                nn.BatchNorm1d(preprocess_hidden_features[i]) if use_batch_norm[i] else nn.Identity(),
                activation_fns[i](),
                nn.Dropout(dropout_rates[i])
            )
            self.atom_preprocess.append(layer)

        self.cheb_convolutions = nn.ModuleList()
        in_channels = preprocess_hidden_features[-1]
        for i in range(len(cheb_hidden_features)):
            self.cheb_convolutions.append(ChebConv(in_channels, cheb_hidden_features[i], K[i], normalization=cheb_normalizations[i]))
            in_channels = cheb_hidden_features[i]

        self.postprocess = nn.ModuleList()
        for i in range(len(postprocess_hidden_features)):
            layer = nn.Sequential(
                nn.Linear(cheb_hidden_features[i-1] if i > 0 else cheb_hidden_features[-1], postprocess_hidden_features[i]),
                nn.BatchNorm1d(postprocess_hidden_features[i]) if use_batch_norm[len(preprocess_hidden_features) + i] else nn.Identity(),
                activation_fns[len(preprocess_hidden_features) + i](),
                nn.Dropout(dropout_rates[len(preprocess_hidden_features) + i])
            )
            self.postprocess.append(layer)

        self.output_layer = nn.Linear(postprocess_hidden_features[-1], out_features)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_preprocess[0](x, edge_index, edge_attr)
        for layer in self.atom_preprocess[1:]:
            x = layer(x)

        for conv in self.cheb_convolutions:
            x = F.relu(conv(x, edge_index))

        for layer in self.postprocess:
            x = layer(x)

        return self.output_layer(x).squeeze(-1)

# %% [markdown]
# ### Гиперпараметры

# %%
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, Timer
import torch.nn as nn

def objective(trial):
    try:
        # Static configuration from the dataset
        in_features = dataset[0].x.shape[1]
        out_features = 1
        edge_attr_dim = dataset[0].edge_attr.shape[1]
        optimizer_class = Lion
        metric = 'rmse'

        # Dynamic parameters to optimize using Optuna
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2,log=True)
        step_size = trial.suggest_int('step_size', 20, 160, step=20)
        gamma = trial.suggest_uniform('gamma', 0.1, 0.9)

        preprocess_hidden_features = [trial.suggest_int('hidden_size', 128, 1024, step=128)] * 9
        postprocess_hidden_features = [trial.suggest_int('hidden_size', 128, 1024, step=128)] * 2

        cheb_hidden_features = [
            trial.suggest_int('hidden_size', 128, 1024, step=128),
            trial.suggest_int('hidden_size', 128, 1024, step=128)
        ]
        cheb_normalization = ['sym', 'sym']
        K = [10, 16]

        dropout_rates = [0.0] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
        activation_fns = [nn.PReLU] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
        use_batch_norm = [True] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))

        batch_size = 512

        backbone = Model(
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
            out_features=out_features
        )

        model = MoleculeModel(
            model_backbone=backbone,
            optimizer_class=optimizer_class,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            step_size=step_size,
            gamma=gamma,
            batch_size=batch_size,
            metric=metric
        )

        data_module = MoleculeDataModule(dataset, batch_size=batch_size, num_workers=8)

        early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        timer = Timer()
        logger = pl.loggers.TensorBoardLogger('tb_logs', name='hyperopt/full')

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



