import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import pytorch_lightning as pl

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class MoleculeModel(pl.LightningModule):
    def __init__(self, model_backbone, optimizer_class, learning_rate, weight_decay, step_size, gamma, batch_size, metric='rmse'):
        super(MoleculeModel, self).__init__()
        self.model_backbone = model_backbone
        self.batch_size = batch_size
        self.metric = self.get_metric(metric)
        
        self.save_hyperparameters(ignore=['model_backbone'])

    def forward(self, x, edge_index, edge_attr):
        return self.model_backbone(x, edge_index, edge_attr)
    
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
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        val_loss = self.metric(batch.y, y_hat)
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)

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