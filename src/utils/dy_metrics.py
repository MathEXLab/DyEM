import numpy as np

def calculate_mse(true, pred, axis = (1)):
    if axis == None:
        return (true - pred)**2
    return np.mean((true - pred)**2, axis = axis)

def calculate_mae(true, pred, axis = (1)):
    if axis == None:
        return np.abs(true - pred)
    return np.mean(np.abs(true - pred), axis = axis)

def calculate_mse_di(di_pred, di_true, axis = (0)):
    if axis == None:
        return (di_pred.squeeze() - di_true.squeeze())**2
    return np.mean((di_pred.squeeze() - di_true.squeeze())**2, axis = axis)

def calculate_mae_di(di_pred, di_true, axis = (0)):
    if axis == None:
        return np.abs(di_pred.squeeze() - di_true.squeeze())
    return np.mean(np.abs(di_pred.squeeze() - di_true.squeeze()), axis = axis)

# normalized version
def calculate_nmse(true, pred,axis = (1)):
    return np.mean((true - pred)**2, axis = axis)/np.var(true)

def calculate_nmae(true, pred,axis = (1)):
    return np.mean(np.abs(true - pred), axis = axis)/np.mean(np.abs(true))

def calculate_nmse_di(di_pred, di_true,axis = (0)):
    if axis == None:
        return (di_pred.squeeze() - di_true.squeeze())**2/np.var(di_true)
    return np.mean((di_pred.squeeze() - di_true.squeeze())**2,axis = axis)/np.var(di_true)

def calculate_nmae_di(di_pred, di_true, axis = (0)):
    if axis == None:
        return np.abs(di_pred.squeeze() - di_true.squeeze())/np.abs(di_true)
    return np.mean(np.abs(di_pred.squeeze() - di_true.squeeze()),axis = axis)/np.mean(np.abs(di_true))
