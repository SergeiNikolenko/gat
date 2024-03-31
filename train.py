# %%
import numpy as np
import pandas as pd
import tensorboard
from rdkit import Chem

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer

from lion_pytorch import Lion

if torch.cuda.is_available():
    print("cuda", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    print("CUDA is not available.")


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")


torch.cuda.empty_cache()

from utils.train import MoleculeModel, MoleculeDataModule, evaluate_model
from utils.prepare import MoleculeData, MoleculeDataset, FeaturizationParameters


# %%
molecule_dataset = torch.load("../data/QM_137k.pt")

# %%
molecule_dataset[0]

# %%
batch_size = 128   
num_workers = 8  

data_module = MoleculeDataModule(molecule_dataset, batch_size=batch_size, num_workers=num_workers)

# %%
in_features = molecule_dataset[0].x.shape[1]
edge_attr_dim = molecule_dataset[0].edge_attr.shape[1]
out_features = 1

hidden_features = [128, 128, 128, 128, 128, 128, 128, 128, 128]  # Размеры предобработки для каждого слоя
postprocess_hidden_features = [128, 128]  # Размеры слоёв постобработки
num_heads = [16, 20]  # Количество голов внимания для каждого слоя GATv2

dropout_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  
activation_fns = [nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU, nn.PReLU]
use_batch_norm = [True, True, True, True, True, True, True, True, True, True, True, True, True]


optimizer_class = Lion

learning_rate = 2.2e-5
weight_decay = 3e-5

step_size = 80
gamma = 0.2

max_epochs = 100
patience = 5

#torch.set_float32_matmul_precision('medium')

model = MoleculeModel(
    atom_in_features=in_features,
    edge_in_features=edge_attr_dim,
    preprocess_hidden_features=hidden_features,
    num_heads=num_heads,
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
logger = pl.loggers.TensorBoardLogger('tb_logs', name='MolModel')

trainer = pl.Trainer(
    max_epochs=max_epochs,
    enable_checkpointing=False,
    callbacks=[early_stop_callback, timer],
    enable_progress_bar=False,
    logger=False,
    accelerator='gpu',
    devices=1,
)

# %%
trainer.fit(model, data_module)

# %%
seconds = timer.time_elapsed()
h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)

print(f"Время обучения: {h}:{m:02d}:{s:02d}")


# %%
evaluate_model(model, data_module)

# %%
def draw_molecule(smiles, predictions):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    predictions_rounded = np.round(predictions, 2)

    for atom, pred in zip(mol.GetAtoms(), predictions_rounded):
        atom.SetProp('atomNote', str(pred))

    img = Chem.Draw.MolToImage(mol, size=(600, 600), kekulize=True)
    img.show()

#smiles = df_results.iloc[0]['smiles']
#predictions = df_results.iloc[0]['predictions']

#draw_molecule(smiles, predictions)



