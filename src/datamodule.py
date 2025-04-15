import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import lightning as L



class ForecastDataset(Dataset):
    def __init__(self, data, init_steps, pred_steps):
        self.data = torch.from_numpy(data)
        self.data = self.data.to(torch.float32)
        self.n_init_steps = init_steps
        self.n_pred_steps = pred_steps
        self.sample_size = self.n_init_steps + self.n_pred_steps
        self.n_samples = len(data) - self.sample_size + 1
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.n_init_steps:idx+self.sample_size, ...]


class FeatureTargetDataset(Dataset):
    def __init__(self, feature, target, ):
        self.feature = torch.from_numpy(feature)
        self.feature = self.feature.to(torch.float32)
        self.target = torch.from_numpy(target)
        self.target = self.target.to(torch.float32)
        self.n_samples = self.feature.shape[0]

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.feature[idx, ...], self.target[idx, ...]

dataset_dict = {    
    'ForecastDataset': ForecastDataset,
    'FeatureTargetDataset': FeatureTargetDataset,
}

# lightning datamodule
class LitDataModule(L.LightningDataModule):
    def __init__(self, cfg,):
        super().__init__()
        self.data_path = cfg.data_path
        self.init_steps = cfg.init_steps
        self.pred_steps = cfg.pred_steps
        self.batch_size = cfg.batch_size
        self.n_workers = cfg.n_workers

        if hasattr(cfg, 'dataset'):
            self.dataset = cfg.dataset
        else:
            self.dataset = 'ForecastDataset'

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.dataset not in dataset_dict:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        elif self.dataset == 'FeatureTargetDataset':
            if stage == 'fit' or stage is None:
                feature = np.load(os.path.join(self.data_path, 'train','feature.npy'))
                target = np.load(os.path.join(self.data_path, 'train','target.npy'))
                self.train_dataset = FeatureTargetDataset(feature, target)
                feature = np.load(os.path.join(self.data_path, 'val','feature.npy'))
                target = np.load(os.path.join(self.data_path, 'val','target.npy'))
                self.val_dataset = FeatureTargetDataset(feature, target)
            if stage == 'test' or stage is None:
                feature = np.load(os.path.join(self.data_path, 'test','feature.npy'))
                target = np.load(os.path.join(self.data_path, 'test','target.npy'))
                self.test_dataset = FeatureTargetDataset(feature, target)

        else:
            # forecast dataset
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                self.train_dataset = ForecastDataset(train_data, self.init_steps, self.pred_steps)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                self.val_dataset = ForecastDataset(self.val_data, self.init_steps, self.pred_steps)
            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.test_dataset = ForecastDataset(self.test_data, self.init_steps, self.pred_steps)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

    def val_dataloader(self):
        # data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
        # self.val_dataset = ForecastDataset(data, self.init_steps, self.pred_steps)
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.n_workers)

    def test_dataloader(self):
        # data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
        # self.test_dataset = ForecastDataset(data, self.init_steps, self.pred_steps)
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
    
if __name__ == '__main__':
    # test dataloader
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/Lorenz_LSTM.yaml")   
    dm = LitDataModule(cfg.data)
    batch = next(iter(dm.train_dataloader()))
    print(batch[0].shape, batch[1].shape)