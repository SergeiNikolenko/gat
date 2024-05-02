# %%
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

from utils.utils import MoleculeDataModule
from utils.train import MoleculeModel

# %%
dataset = torch.load(f'../data/QM_137k.pt')
# %%
in_features = dataset[0].x.shape[1]
out_features = 1
edge_attr_dim = dataset[0].edge_attr.shape[1]

preprocess_hidden_features = [128] * 9
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

dropout_rates = [0.0] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
activation_fns = [nn.PReLU] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
use_batch_norm = [True] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))

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
logger = pl.loggers.TensorBoardLogger('tb_logs', name='ChebConvEdge')

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



