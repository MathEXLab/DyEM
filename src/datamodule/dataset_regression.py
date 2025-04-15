import torch
import numpy as np
from torch.utils.data import Dataset




class RegDeepONetDataset(Dataset): 
    def __init__(self, data, init_steps, pred_steps, return_last = False, predict_index:list = None, coor_path:str = None):
        super().__init__()
        '''
        This dataset class is used to create a dataset for forecasting tasks, using sliding window approach.
        Args:
            data: numpy array of shape (n_samples, ..., n_features)
            init_steps: int, number of initial steps to be used for forecasting
            pred_steps: int, number of steps to be forecasted
            return_last: bool, if True, only return the last day

        
        Returns:
            x: torch tensor of shape (init_steps, ..., n_features)
            y: torch tensor of shape (pred_steps, ..., n_features)
        
        '''
        self.data = torch.from_numpy(data)
        self.data = self.data.to(torch.float32)

        self.n_init_steps = init_steps
        self.n_pred_steps = pred_steps

        self.sample_size = self.n_init_steps + self.n_pred_steps
        
        self.n_samples = len(data) - self.sample_size + 1

        self.return_last = return_last  # if return last, only return the last day
        self.predict_index = predict_index # if provided, only send the index in the list as target

        # loc, for deeponet
        self.loc = torch.from_numpy(np.load(coor_path))
        self.loc = self.loc.to(torch.float32)

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if self.return_last:
            if self.predict_index:
                return (self.data[idx:idx+self.n_init_steps, ...],self.loc), self.data[idx+self.sample_size-1, ...][None,...][...,self.predict_index]
            else:
                return (self.data[idx:idx+self.n_init_steps, ...],self.loc), self.data[idx+self.sample_size-1, ...][None,...]  # last step
        else:
            if self.predict_index:
                return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.n_init_steps:idx+self.sample_size, ...][...,self.predict_index]
            return (self.data[idx:idx+self.n_init_steps, ...],self.loc), self.data[idx+self.n_init_steps:idx+self.sample_size, ...]




class RegTargetDataset(Dataset):

    def __init__(self, feature, target,):
        super().__init__()
        '''
        This dataset class is used to create a dataset for forecasting tasks, using feature and target.
        The feature and taget should have same dimension in the first dimension
        Args:
            feature: numpy array of shape (n_samples, ..., n_features)
            target: numpy array of shape (n_samples, ..., n_features)

        Returns:
        '''
        self.feature = torch.from_numpy(feature)
        self.feature = self.feature.to(torch.float32)
        self.target = torch.from_numpy(target)
        self.target = self.target.to(torch.float32)

    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        return self.feature[idx,...], self.target[idx,...]
    
class RegAEDataset(Dataset):    # autoencoder

    def __init__(self, feature,):
        super().__init__()
        '''
        This dataset class is used to create a dataset for autoencoder
        Args:
            feature: numpy array of shape (n_samples, ..., n_features)

        Returns:
        '''
        self.feature = torch.from_numpy(feature)
        self.feature = self.feature.to(torch.float32)

    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        return self.feature[idx,None,...], self.feature[idx,None,...]   #(n_samples, SEQ_LEN, n_features)


class RegSlidingDataset(Dataset):

    def __init__(self, data, init_steps, pred_steps, return_last = False, predict_index:list = None,weights=None):
        super().__init__()
        '''
        This dataset class is used to create a dataset for forecasting tasks, using sliding window approach.
        Args:
            data: numpy array of shape (n_samples, ..., n_features)
            init_steps: int, number of initial steps to be used for forecasting
            pred_steps: int, number of steps to be forecasted
            return_last: bool, if True, only return the last day

        
        Returns:
            x: torch tensor of shape (init_steps, ..., n_features)
            y: torch tensor of shape (pred_steps, ..., n_features)
        
        '''
        self.data = torch.from_numpy(data)
        self.data = self.data.to(torch.float32)

        self.n_init_steps = init_steps
        self.n_pred_steps = pred_steps

        self.sample_size = self.n_init_steps + self.n_pred_steps
        
        self.n_samples = len(data) - self.sample_size + 1

        self.return_last = return_last  # if return last, only return the last day
        self.predict_index = predict_index # if provided, only send the index in the list as target

        self.use_weights = False
        if weights is not None:
            self.weights = torch.from_numpy(weights)
            self.weights = self.weights.to(torch.float32)
            self.use_weights = True

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if self.return_last:
            if self.predict_index:
                if self.use_weights:
                    return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.sample_size-1, ...][None,...][...,self.predict_index], self.weights[idx+self.sample_size-1, ...][None,...][...,self.predict_index]
                else:
                    return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.sample_size-1, ...][None,...][...,self.predict_index]
            else:
                return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.sample_size-1, ...][None,...]  # last step
        else:
            if self.predict_index:
                if self.use_weights:
                    return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.n_init_steps:idx+self.sample_size, ...][...,self.predict_index], self.weights[idx+self.n_init_steps:idx+self.sample_size, ...][...,self.predict_index]
                else:
                    return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.n_init_steps:idx+self.sample_size, ...][...,self.predict_index]
            return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.n_init_steps:idx+self.sample_size, ...]


class DyEmbRegDataset(Dataset):

    def __init__(self, data,d,theta, init_steps, pred_steps, return_last = False, predict_index:list = None,weights=None,pre_compute = False):
        super().__init__()
        '''
        This dataset class is used to create a dataset for forecasting tasks, using sliding window approach.
        Args:
            data: numpy array of shape (n_samples, ..., n_features)
            init_steps: int, number of initial steps to be used for forecasting
            pred_steps: int, number of steps to be forecasted
            return_last: bool, if True, only return the last day

        Returns:
            x: torch tensor of shape (init_steps, ..., n_features)
            y: torch tensor of shape (pred_steps, ..., n_features)
            d: torch tensor of shape (n_samples, 1)
            theta: torch tensor of shape (n_samples, 1)
        
        '''

        self.data = torch.from_numpy(data)
        self.data = self.data.to(torch.float32)
        if pre_compute:
            self.pre_compute = True
            self.d = torch.from_numpy(d.squeeze().reshape(-1,1))
            self.d = self.d.to(torch.float32)
            self.theta = torch.from_numpy(theta.squeeze().reshape(-1,1))
            self.theta = self.theta.to(torch.float32)

        self.n_init_steps = init_steps
        self.n_pred_steps = pred_steps

        self.sample_size = self.n_init_steps + self.n_pred_steps
        
        self.n_samples = len(data) - self.sample_size + 1

        self.return_last = return_last  # if return last, only return the last day
        self.predict_index = predict_index # if provided, only send the index in the list as target

        self.use_weights = False
        if weights is not None:
            self.weights = torch.from_numpy(weights)
            self.weights = self.weights.to(torch.float32)
            self.use_weights = True

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if self.pre_compute:
            return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.sample_size-1, ...][None,...], self.d[idx+self.sample_size-1, ...][None,...], self.theta[idx+self.sample_size-1, ...][None,...]
        else:
            return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.sample_size-1, ...][None,...]  # last step
        

class RegWeightedDataset(Dataset):
    def __init__(self, data=None,weights=None, init_steps=None, pred_steps=None, **kwargs):
        super().__init__()
        '''
        This dataset class is used to create a dataset for forecasting tasks, using sliding window approach.
        precomputed weights are used
        Args:
            data: numpy array of shape (n_samples, ..., n_features)
            init_steps: int, number of initial steps to be used for forecasting
            pred_steps: int, number of steps to be forecasted
            weights: numpy array of shape (n_samples, ..., n_features)
        '''
        self.data = torch.from_numpy(data)
        self.data = self.data.to(torch.float32)
        if weights is not None:
            self.weights = torch.from_numpy(weights)
            self.weights = self.weights.to(torch.float32)
            self.use_weights = True
        else:
            self.use_weights = False

        self.n_init_steps = init_steps
        self.n_pred_steps = pred_steps

        self.sample_size = self.n_init_steps + self.n_pred_steps
        self.n_samples = len(data) - self.sample_size + 1

        self.return_last = kwargs.get('return_last', False)

        self.kwargs = kwargs

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.return_last:
            if self.use_weights:
                return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.sample_size-1, ...][None,...], self.weights[idx+self.sample_size-1, ...][None,...]
            else:
                return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.sample_size-1, ...][None,...]
        else:
            if self.use_weights:
                return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.n_init_steps:idx+self.sample_size, ...], self.weights[idx+self.n_init_steps:idx+self.sample_size, ...]
            else:
                return self.data[idx:idx+self.n_init_steps, ...], self.data[idx+self.n_init_steps:idx+self.sample_size, ...]