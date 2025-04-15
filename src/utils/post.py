import os
import torch
import numpy as np

def get_prediction(result_dir, version = 0):
    result_dir = os.path.join(result_dir, 'version_{}'.format(version))
    prediction = torch.load(os.path.join(result_dir, 'checkpoints','predictions.pt'))
    pred = []
    true = []
    for i in range(len(prediction)):
        y_hat, y = prediction[i]
        pred.append(y_hat.detach().cpu().numpy())
        true.append(y.detach().cpu().numpy())

    # cocatenate all the predictions and true values in axis 0
    pred = np.concatenate(pred, axis=0).squeeze()
    true = np.concatenate(true, axis=0).squeeze()

    # print(pred.shape, true.shape)

    mse_series = ((pred - true)**2).mean(axis = -1)
    # print(mse_series.mean(axis = 0))
    return pred, true, mse_series

def get_mse_series_mean_std(result_dir, versions = [0,1,2]):
    all_mse = []
    for v in versions:
        _,_,mse_series = get_prediction(result_dir, v)
        all_mse.append(mse_series)
    all_mse = np.array(all_mse).T
    mean = all_mse.mean(axis = 1)
    std = all_mse.std(axis = 1)
    return mean, std
