# %%
import pandas as pd
import numpy as np

import time

from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer

from torch_geometric.nn import GATv2Conv, TransformerConv
from torch_scatter import scatter_mean

from lion_pytorch import Lion

import matplotlib.pyplot as plt


if torch.cuda.is_available():
    print("cuda", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    print("CUDA is not available.")


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from utils.train import MoleculeModel, MoleculeDataModule, get_metric, GATv2Model
from utils.prepare import MoleculeData, MoleculeDataset, FeaturizationParameters

# %%
molecule_dataset = torch.load("../../data/QM_137k.pt")

# %%
molecule_dataset[0]

# %%
batch_size = 128   
num_workers = 8  

data_module = MoleculeDataModule(molecule_dataset, batch_size=batch_size, num_workers=num_workers)

# %%
class GATv2Model(nn.Module):
    def __init__(self, atom_in_features, edge_in_features, hidden_features, out_features, num_heads, dropout_rate, activation_fn):
        super(GATv2Model, self).__init__()

        self.atom_preprocess = nn.Sequential(
            nn.Linear(atom_in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            activation_fn,
            nn.Dropout(dropout_rate)
        )

        self.edge_preprocess = nn.Sequential(
            nn.Linear(edge_in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            activation_fn,
            nn.Dropout(dropout_rate)
        )

        self.atom_message_layer = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            activation_fn,
            nn.Dropout(dropout_rate)
        )

        # Первый слой GATv2Conv
        self.gat_conv1 = GATv2Conv(
            in_channels=hidden_features * 2,  # Учитываем атомные сообщения
            out_channels=hidden_features,
            heads=num_heads,
            dropout=dropout_rate,
            concat=True
        )

        # Второй слой GATv2Conv
        self.gat_conv2 = GATv2Conv(
            in_channels=hidden_features * num_heads,  # Выход с предыдущего слоя GATv2
            out_channels=hidden_features,
            heads=num_heads,
            dropout=dropout_rate,
            concat=True
        )

        self.bn1 = nn.BatchNorm1d(hidden_features * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_features * num_heads)

        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout_rate)

        # FFN
        self.postprocess = nn.Sequential(
            nn.Linear(hidden_features * num_heads, hidden_features * 2),
            nn.BatchNorm1d(hidden_features * 2),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features * 2, hidden_features),
            nn.BatchNorm1d(hidden_features),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x, edge_index, edge_attr):
        atom_features = self.atom_preprocess(x)
        edge_features = self.edge_preprocess(edge_attr)

        row, col = edge_index
        agg_edge_features = scatter_mean(edge_features, col, dim=0, dim_size=x.size(0))
        atom_messages = self.atom_message_layer(atom_features + agg_edge_features)

        combined_features = torch.cat([atom_messages, agg_edge_features], dim=1)

        combined_features = self.gat_conv1(combined_features, edge_index)
        combined_features = self.bn1(combined_features)
        combined_features = self.activation(combined_features)
        combined_features = self.dropout(combined_features)

        combined_features = self.gat_conv2(combined_features, edge_index)
        combined_features = self.bn2(combined_features)
        combined_features = self.activation(combined_features)
        combined_features = self.dropout(combined_features)

        out = self.postprocess(combined_features).squeeze(-1)
        return out


# %%
in_features = molecule_dataset[0].x.shape[1]
hidden_features = 64
num_heads = 8

edge_attr_dim = molecule_dataset[0].edge_attr.shape[1]

dropout_rate = 0.0
activation_fn = nn.ReLU()

optimizer_class = Lion 
learning_rate = 0.00085           
weight_decay = 2e-4         

step_size = 50       
gamma = 0.1                     

max_epochs = 100     
patience = 5       

base_model = GATv2Model(
    atom_in_features=in_features,
    hidden_features=hidden_features,
    out_features=1,
    num_heads=num_heads,
    dropout_rate=dropout_rate,
    activation_fn=activation_fn,
    edge_in_features=edge_attr_dim
)


model = MoleculeModel(
    base_model=base_model,
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
logger = pl.loggers.CSVLogger('logs', name='GATv2')
timer = Timer()


trainer = pl.Trainer(
    max_epochs=max_epochs,
    enable_checkpointing = False,
    auto_scale_batch_size=True,
    accelerator='auto',
    devices=1,
    callbacks=[early_stop_callback, timer],
    enable_progress_bar=False,
    logger=False
)

# %%
trainer.fit(model, data_module)

# %%
print(f"Время обучения: {timer.time_elapsed()} секунд")

# %%
df = trainer.test(model, data_module.test_dataloader())

# %%
df_results = model.df_results

all_predictions = np.concatenate(df_results['predictions'].values)
all_true_values = np.concatenate(df_results['true_values'].values)

rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))

print(f"Общий RMSE: {rmse}")

# %%
train_losses = model.train_losses
val_losses = model.val_losses

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.legend()
plt.show()

# %%
def draw_molecule(smiles, predictions):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    predictions_rounded = np.round(predictions, 2)

    for atom, pred in zip(mol.GetAtoms(), predictions_rounded):
        atom.SetProp('atomNote', str(pred))

    img = Chem.Draw.MolToImage(mol, size=(600, 600), kekulize=True)
    img.show()

smiles = df_results.iloc[0]['smiles']
predictions = df_results.iloc[0]['predictions']

draw_molecule(smiles, predictions)



