
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
dataset = torch.load(f'../data/QM_137k.pt')

# %%
#dataset = add_skipatom_features_to_dataset(dataset, min_count=2e7, top_n=4, device='cpu', progress_bar=True, scaler=StandardScaler())

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


class MoleculeModel(pl.LightningModule):
    def __init__(self, atom_in_features, edge_attr_dim, preprocess_hidden_features, cheb_hidden_features, K, cheb_normalizations, dropout_rates, activation_fns, use_batch_norm, postprocess_hidden_features, out_features, optimizer_class, learning_rate, weight_decay, step_size, gamma, batch_size, metric='rmse'):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.metric = self.get_metric(metric)
        
        self.train_losses = []
        self.val_losses = []

        # Preprocessing layers for atom features
        self.atom_preprocess = nn.ModuleList()
        self.atom_preprocess.append(AtomEdgeInteraction(atom_in_features, edge_attr_dim, preprocess_hidden_features[0]))
        for i in range(1, len(preprocess_hidden_features)):
            preprocess_layer = nn.Sequential()
            in_features = preprocess_hidden_features[i-1]
            preprocess_layer.add_module(f'atom_linear_{i}', nn.Linear(in_features, preprocess_hidden_features[i]))
            if use_batch_norm[i]:
                preprocess_layer.add_module(f'atom_bn_{i}', nn.BatchNorm1d(preprocess_hidden_features[i]))
            preprocess_layer.add_module(f'atom_activation_{i}', activation_fns[i]())
            preprocess_layer.add_module(f'atom_dropout_{i}', nn.Dropout(dropout_rates[i]))
            self.atom_preprocess.append(preprocess_layer)

        # Chebyshev convolution layers
        self.cheb_convolutions = nn.ModuleList()
        in_channels = preprocess_hidden_features[-1]
        cheb_index = len(preprocess_hidden_features)
        for i, out_channels in enumerate(cheb_hidden_features):
            normalization = cheb_normalizations[i] if i < len(cheb_normalizations) else 'none'
            self.cheb_convolutions.append(ChebConv(in_channels=in_channels, out_channels=out_channels, K=K[i], normalization=normalization))

            self.cheb_convolutions.append(activation_fns[cheb_index + i]())
            in_channels = out_channels

        # Postprocessing layers
        self.postprocess = nn.ModuleList()
        in_features = cheb_hidden_features[-1]
        post_index = cheb_index + len(cheb_hidden_features)
        for i in range(len(postprocess_hidden_features)):
            post_layer = nn.Sequential()
            post_layer.add_module(f'post_linear_{i}', nn.Linear(in_features, postprocess_hidden_features[i]))
            if use_batch_norm[post_index + i]:
                post_layer.add_module(f'post_bn_{i}', nn.BatchNorm1d(postprocess_hidden_features[i]))
            post_layer.add_module(f'post_activation_{i}', activation_fns[post_index + i]())
            post_layer.add_module(f'post_dropout_{i}', nn.Dropout(dropout_rates[post_index + i]))
            self.postprocess.append(post_layer)
            in_features = postprocess_hidden_features[i]

        self.output_layer = nn.Linear(postprocess_hidden_features[-1], out_features)


    def forward(self, x, edge_index, edge_attr):

        x = self.atom_preprocess[0](x, edge_index, edge_attr)

        for layer in self.atom_preprocess[1:]:
            x = layer(x)

        for idx in range(0, len(self.cheb_convolutions), 2):
            x = self.cheb_convolutions[idx](x, edge_index)
            x = self.cheb_convolutions[idx+1](x)

        for layer in self.postprocess:
            x = layer(x)

        x = self.output_layer(x).squeeze(-1)
        return x
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_class(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def on_train_start(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(self.log_activations_hook(name))

    def training_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        loss = self.metric(batch.y, y_hat)
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        self.train_losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        val_loss = self.metric(batch.y, y_hat)
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
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
    
    def on_epoch_end(self):
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)
            
    def log_activations_hook(self, layer_name):
        def hook(module, input, output):
            if self.logger:
                self.logger.experiment.add_histogram(f"{layer_name}_activations", output, self.current_epoch)
        return hook

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

# %%
in_features = dataset[0].x.shape[1]
out_features = 1
edge_attr_dim = dataset[0].edge_attr.shape[1]

preprocess_hidden_features = [1700] * 9
postprocess_hidden_features = [128, 128]

cheb_hidden_features = [128, 128]
K = [10, 16]
cheb_normalization = ['sym', 'sym']

optimizer_class = Lion
learning_rate = 2.2e-5
weight_decay = 3e-5
step_size = 80
gamma = 0.2
max_epochs = 100
patience = 5
batch_size = 128

batch_size = 128
num_workers = 8  

data_module = MoleculeDataModule(dataset, batch_size=batch_size, num_workers=num_workers)

dropout_rates = [0.0] * (len(preprocess_hidden_features) + len(postprocess_hidden_features) + len(cheb_hidden_features))
activation_fns = [nn.PReLU] * (len(preprocess_hidden_features) + len(postprocess_hidden_features) + len(cheb_hidden_features))
use_batch_norm = [True] * (len(preprocess_hidden_features) + len(postprocess_hidden_features) + len(cheb_hidden_features))

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
    optimizer_class=optimizer_class,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    step_size=step_size,
    gamma=gamma,
    batch_size=batch_size,
    metric='rmse'
)



print("Model:\n", model)

checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=patience, verbose=True, mode='min')
timer = Timer()
logger = pl.loggers.TensorBoardLogger('tb_logs', name='ChebConvPrely')

trainer = pl.Trainer(
    max_epochs=max_epochs,
    enable_checkpointing=False,
    callbacks=[early_stop_callback, timer],
    enable_progress_bar=False,
    logger=logger,
    accelerator='gpu',
    devices=1,
)
trainer.fit(model, data_module)

# %%
seconds = timer.time_elapsed()
h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)

print(f"Время обучения: {h}:{m:02d}:{s:02d}")



