import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def get_metric(metric_name):
    """
    Возвращает функцию метрики на основе ее имени.

    Args:
    metric_name (str): Имя метрики. Допустимые значения: 'mse', 'rmse'.

    Returns:
    function: Функция метрики, которая принимает два аргумента: y_true и y_pred.
    """
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

class MoleculeDataModule(pl.LightningDataModule):
    """Модуль данных для обработки набора данных молекул.

    Args:
        dataset (Dataset): Набор данных, содержащий молекулы.
        batch_size (int, optional): Размер батча для обучения. По умолчанию 128.
        val_split (float, optional): Доля набора данных, выделенная для валидации. По умолчанию 0.1.
        test_split (float, optional): Доля набора данных, выделенная для тестирования. По умолчанию 0.2.
        num_workers (int, optional): Количество рабочих процессов для загрузки данных. По умолчанию 1.
    """
    def __init__(self, dataset, batch_size=128, val_split=0.1, test_split=0.2, num_workers=1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        """Настройка разделов датасета на тренировочный, валидационный и тестовый."""
        indices = list(range(len(self.dataset)))
        train_val_indices, test_indices = train_test_split(indices, test_size=self.test_split, random_state=42)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=self.val_split / (1 - self.test_split), random_state=42)
        
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
    
    def train_dataloader(self):
        """Создает DataLoader для тренировочных данных."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        """Создает DataLoader для валидационных данных."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        """Создает DataLoader для тестовых данных."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



class MoleculeModel(pl.LightningModule):
    """Модель для предсказания свойств молекул.

    Args:
        base_model (nn.Module): Базовая модель (например, GNN).
        optimizer_class (Optimizer): Класс оптимизатора.
        learning_rate (float): Скорость обучения.
        weight_decay (float): Величина L2 регуляризации.
        step_size (int): Частота обновления скорости обучения.
        gamma (float): Множитель для скорости обучения.
        batch_size (int): Размер батча.
    """
    def __init__(self, base_model, optimizer_class, learning_rate, weight_decay, step_size, gamma, batch_size, metric='rmse'):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])
        self.base_model = base_model
        self.batch_size = batch_size
        self.metric = get_metric(metric)

        self.train_losses = []
        self.val_losses = []

    def forward(self, x, edge_index):
        """Выполняет прямой проход модели."""
        return self.base_model(x, edge_index)

    def configure_optimizers(self):
        """Настройка оптимизатора и планировщика скорости обучения."""
        optimizer = self.hparams.optimizer_class(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    
    '''    def training_step(self, batch, batch_idx):
            """Один шаг обучения."""
            y_hat = self(batch.x, batch.edge_index)
            loss = F.mse_loss(y_hat, batch.y)
            self.log('train_loss', loss, batch_size=self.batch_size)
            self.train_losses.append(loss.item())
            return loss
    '''

    def training_step(self, batch, batch_idx):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=tensorboard_trace_handler('profiler')
        ) as prof:
            y_hat = self(batch.x, batch.edge_index)
            loss = self.metric(batch.y, y_hat)
            self.log('train_loss', loss, batch_size=self.batch_size)
            self.train_losses.append(loss.item())
        
        prof.step()

        return loss
    

    def validation_step(self, batch, batch_idx):
        """Один шаг валидации."""
        y_hat = self(batch.x, batch.edge_index)
        val_loss = self.metric(batch.y, y_hat)
        self.log('val_loss', val_loss, batch_size=self.batch_size)
        self.val_losses.append(val_loss.item())

    def test_step(self, batch, batch_idx):
        """Один шаг тестирования."""
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


    def r2_score(self, y_true, y_pred):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def test_epoch_end(self, outputs):
        """Обработка результатов после завершения тестирования."""
        all_data = [item for batch_data in outputs for item in batch_data]
        self.df_results = pd.DataFrame(all_data)

        all_predictions = np.concatenate(self.df_results['predictions'].values)
        all_true_values = np.concatenate(self.df_results['true_values'].values)

        rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
        mse = mean_squared_error(all_true_values, all_predictions)
        r2 = r2_score(all_true_values, all_predictions)
        mae = mean_absolute_error(all_true_values, all_predictions)

        print(f'RMSE: {rmse:.4f}')
        print(f'MSE: {mse:.4f}')
        print(f'R²: {r2:.4f}')
        print(f'MAE: {mae:.4f}')

        return self.df_results