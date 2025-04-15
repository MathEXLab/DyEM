import torch
import lightning as L
from typing import Any, Dict, Tuple
import numpy as np
import os

from src.datamodule.dataset_regression import RegSlidingDataset, RegTargetDataset, RegDeepONetDataset, RegAEDataset, DyEmbRegDataset, RegWeightedDataset    
        


data_type_dict = {'sliding': RegSlidingDataset,
                  'target': RegTargetDataset,
                  'deeponet': RegDeepONetDataset,
                  'ae': RegAEDataset,
                  'dyemb': DyEmbRegDataset,
                  'weighted': RegWeightedDataset}



# lightning datamodule
class RegLitDataModule(L.LightningDataModule):
    def __init__(self, cfg,):
        super().__init__()
        '''
        This class is used to create a lightning datamodule for forecasting tasks.
        Args:
            cfg: the 'data' part in config file
        '''
        self.data_path = cfg.data_path  # path to the data directory
        if hasattr(cfg, 'init_steps'):
            self.init_steps = cfg.init_steps
        if hasattr(cfg, 'pred_steps'):
            self.pred_steps = cfg.pred_steps

        self.batch_size = cfg.batch_size    # batch size
        self.n_workers = cfg.n_workers    # number of workers for dataloader

        if hasattr(cfg, 'return_last'):
            self.return_last = cfg.return_last
        else:
            self.return_last = False
        
        if hasattr(cfg, 'predict_index'):
            self.predict_index = cfg.predict_index
        else:
            self.predict_index = None

        if hasattr(cfg, 'data_type'):
            self.data_type = cfg.data_type
        else:
            self.data_type = 'sliding'

        if self.data_type == 'deeponet':
            assert hasattr(cfg, 'coor_path'), 'coor_path should be provided for deeponet'
            self.coor_path = cfg.coor_path

        if self.data_type == 'sliding':
            assert hasattr(cfg, 'init_steps') and hasattr(cfg, 'pred_steps'), 'init_steps and pred_steps should be provided for sliding data'

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if self.data_type == 'sliding':
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                self.train_dataset = data_type_dict[self.data_type](train_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                self.val_dataset = data_type_dict[self.data_type](self.val_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.test_dataset = data_type_dict[self.data_type](self.test_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
            
            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.predict_dataset = data_type_dict[self.data_type](self.predict_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)

        elif self.data_type == 'target':
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'feature.npy')) 
                train_target = np.load(os.path.join(self.data_path, 'train', 'target.npy')) 
                self.train_dataset = RegTargetDataset(train_data, train_target)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'feature.npy'))
                self.val_target = np.load(os.path.join(self.data_path, 'val', 'target.npy'))
                self.val_dataset = RegTargetDataset(self.val_data, self.val_target)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'feature.npy'))
                self.test_target = np.load(os.path.join(self.data_path, 'test', 'target.npy'))
                self.test_dataset = RegTargetDataset(self.test_data, self.test_target)

            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'feature.npy'))
                self.predict_target = np.load(os.path.join(self.data_path, 'test', 'target.npy'))
                self.predict_dataset = RegTargetDataset(self.predict_data, self.predict_target)

        elif self.data_type == 'deeponet':
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                self.train_dataset = data_type_dict[self.data_type](train_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index, self.coor_path)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                self.val_dataset = data_type_dict[self.data_type](self.val_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index, self.coor_path)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.test_dataset = data_type_dict[self.data_type](self.test_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index, self.coor_path)
            
            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.predict_dataset = data_type_dict[self.data_type](self.predict_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index, self.coor_path)
            
        elif self.data_type == 'ae':
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                self.train_dataset = data_type_dict[self.data_type](train_data)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                self.val_dataset = data_type_dict[self.data_type](self.val_data)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.test_dataset = data_type_dict[self.data_type](self.test_data)
            
            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.predict_dataset = data_type_dict[self.data_type](self.predict_data)

        else:
            raise ValueError(f"Data type {self.data_type} not supported")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
    
    def re_train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.n_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
    

# lightning datamodule
class DyEmbRegDataModule(L.LightningDataModule):
    def __init__(self, cfg,):
        super().__init__()
        '''
        This class is used to create a lightning datamodule for forecasting tasks.
        Args:
            cfg: the 'data' part in config file
        '''
        self.data_path = cfg.data_path  # path to the data directory
        if hasattr(cfg, 'init_steps'):
            self.init_steps = cfg.init_steps
        if hasattr(cfg, 'pred_steps'):
            self.pred_steps = cfg.pred_steps

        self.batch_size = cfg.batch_size    # batch size
        self.n_workers = cfg.n_workers    # number of workers for dataloader

        if hasattr(cfg, 'return_last'):
            self.return_last = cfg.return_last
        else:
            self.return_last = False
        
        if hasattr(cfg, 'predict_index'):
            self.predict_index = cfg.predict_index
        else:
            self.predict_index = None

        if hasattr(cfg, 'data_type'):
            self.data_type = cfg.data_type
        else:
            self.data_type = 'sliding'

        if self.data_type == 'deeponet':
            assert hasattr(cfg, 'coor_path'), 'coor_path should be provided for deeponet'
            self.coor_path = cfg.coor_path

        if self.data_type == 'sliding':
            assert hasattr(cfg, 'init_steps') and hasattr(cfg, 'pred_steps'), 'init_steps and pred_steps should be provided for sliding data'

        if self.data_type == 'dyemb':
            assert hasattr(cfg, 'base_attractor'), 'base_attractor should be provided for dyemb data'

        if hasattr(cfg, 'use_precomputed_di') and cfg.use_precomputed_di:
            self.use_precomputed_di = True

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if self.data_type == 'sliding':
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                self.train_dataset = data_type_dict[self.data_type](train_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                self.val_dataset = data_type_dict[self.data_type](self.val_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.test_dataset = data_type_dict[self.data_type](self.test_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
            
            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.predict_dataset = data_type_dict[self.data_type](self.predict_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)

        elif self.data_type == 'target':
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'feature.npy')) 
                train_target = np.load(os.path.join(self.data_path, 'train', 'target.npy')) 
                self.train_dataset = RegTargetDataset(train_data, train_target)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'feature.npy'))
                self.val_target = np.load(os.path.join(self.data_path, 'val', 'target.npy'))
                self.val_dataset = RegTargetDataset(self.val_data, self.val_target)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'feature.npy'))
                self.test_target = np.load(os.path.join(self.data_path, 'test', 'target.npy'))
                self.test_dataset = RegTargetDataset(self.test_data, self.test_target)

            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'feature.npy'))
                self.predict_target = np.load(os.path.join(self.data_path, 'test', 'target.npy'))
                self.predict_dataset = RegTargetDataset(self.predict_data, self.predict_target)

        elif self.data_type == 'deeponet':
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                self.train_dataset = data_type_dict[self.data_type](train_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index, self.coor_path)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                self.val_dataset = data_type_dict[self.data_type](self.val_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index, self.coor_path)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.test_dataset = data_type_dict[self.data_type](self.test_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index, self.coor_path)
            
            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.predict_dataset = data_type_dict[self.data_type](self.predict_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index, self.coor_path)
            
        elif self.data_type == 'ae':
            if stage == 'fit' or stage is None:
                train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                self.train_dataset = data_type_dict[self.data_type](train_data)
                self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                self.val_dataset = data_type_dict[self.data_type](self.val_data)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.test_dataset = data_type_dict[self.data_type](self.test_data)
            
            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                self.predict_dataset = data_type_dict[self.data_type](self.predict_data)
        
        elif self.data_type == 'dyemb':
            if stage == 'fit' or stage is None:
                if self.use_precomputed_di:
                    train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                    train_d = np.load(os.path.join(self.data_path, 'train', 'd.npy'))
                    train_theta = np.load(os.path.join(self.data_path, 'train', 'theta.npy'))
                    self.train_dataset = data_type_dict[self.data_type](train_data, train_d, train_theta, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
                    self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                    self.val_d = np.load(os.path.join(self.data_path, 'val', 'd.npy'))
                    self.val_theta = np.load(os.path.join(self.data_path, 'val', 'theta.npy'))
                    self.val_dataset = data_type_dict[self.data_type](self.val_data, self.val_d, self.val_theta, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
                else:
                    train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
                    self.train_dataset = data_type_dict[self.data_type](train_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
                    self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
                    self.val_dataset = data_type_dict[self.data_type](self.val_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)

            if stage == 'test' or stage is None:
                self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                if self.use_precomputed_di:
                    self.test_d = np.load(os.path.join(self.data_path, 'test', 'd.npy'))
                    self.test_theta = np.load(os.path.join(self.data_path, 'test', 'theta.npy'))
                    self.test_dataset = data_type_dict[self.data_type](self.test_data, self.test_d, self.test_theta, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
                else:
                    self.test_dataset = data_type_dict[self.data_type](self.test_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
            
            if stage == 'predict' or stage is None:
                self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                if self.use_precomputed_di:
                    self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
                    self.predict_d = np.load(os.path.join(self.data_path, 'test', 'd.npy'))
                    self.predict_theta = np.load(os.path.join(self.data_path, 'test', 'theta.npy'))
                    self.predict_dataset = data_type_dict[self.data_type](self.predict_data, self.predict_d, self.predict_theta, self.init_steps, self.pred_steps, self.return_last, self.predict_index)
                else:
                    self.predict_dataset = data_type_dict[self.data_type](self.predict_data, self.init_steps, self.pred_steps, self.return_last, self.predict_index)

        else:
            raise ValueError(f"Data type {self.data_type} not supported")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.n_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.n_workers)



class WeightedDataModule(L.LightningDataModule):
    def __init__(self, cfg,):
        super().__init__()
        '''
        This class is used to create a lightning datamodule for forecasting tasks.
        Args:
            cfg: the 'data' part in config file
        '''
        self.data_path = cfg.data_path  # path to the data directory
        if hasattr(cfg, 'init_steps'):
            self.init_steps = cfg.init_steps
        if hasattr(cfg, 'pred_steps'):
            self.pred_steps = cfg.pred_steps

        self.batch_size = cfg.batch_size    # batch size
        self.n_workers = cfg.n_workers    # number of workers for dataloader

        if hasattr(cfg, 'return_last'):
            self.return_last = cfg.return_last
        else:
            self.return_last = False

        self.data_type = cfg.data_type

        if hasattr(cfg, 'weights_type'):
            self.weights_type = cfg.weights_type
        else:
            self.weights_type = 'linear'

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print('using weighted datamodule')
        if stage == 'fit' or stage is None:
            train_data = np.load(os.path.join(self.data_path, 'train', 'data.npy')) 
            train_weight = np.load(os.path.join(self.data_path, 'train', self.weights_type +'_weights.npy'))
            self.train_dataset = data_type_dict[self.data_type](data =train_data, weights=train_weight,init_steps= self.init_steps,
                                                                pred_steps=self.pred_steps, kwargs = {'return_last':self.return_last})
            self.val_data = np.load(os.path.join(self.data_path, 'val', 'data.npy'))
            # self.val_dataset = data_type_dict[self.data_type](data = self.val_data, weights=None,init_steps= self.init_steps,
            #                                                     pred_steps=self.pred_steps, kwargs = {'return_last':self.return_last})
            self.val_dataset = data_type_dict[self.data_type](data = self.val_data, weights=train_weight,init_steps= self.init_steps,
                                                        pred_steps=self.pred_steps, kwargs = {'return_last':self.return_last})

        if stage == 'test' or stage is None:
            self.test_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
            self.test_dataset = data_type_dict[self.data_type](data = self.test_data, weights=None, init_steps=self.init_steps,
                                                                pred_steps=self.pred_steps, kwargs = {'return_last':self.return_last})
        
        if stage == 'predict' or stage is None:
            self.predict_data = np.load(os.path.join(self.data_path, 'test', 'data.npy'))
            self.predict_dataset = data_type_dict[self.data_type](data = self.test_data, weights=None, init_steps=self.init_steps,
                                                                pred_steps=self.pred_steps, kwargs = {'return_last':self.return_last})


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.n_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
    
if __name__ == '__main__':
    # test dataloader
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("/home/Extremes-Classification/configs/TestML_regression.yaml")   
    dm = RegLitDataModule(cfg.data)
    dm.setup('fit')
    batch = next(iter(dm.train_dataloader()))
    print(batch[0].shape, batch[1].shape)