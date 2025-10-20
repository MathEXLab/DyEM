import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from tqdm import tqdm
import time

from src.pypardi.di_evaluate import compute as li_ar



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='train/data.npy')   # data as base attractor
    parser.add_argument('--new', type=str, default='test/data.npy') # new data to compute DI against base attractor, e.g. ML forecast
    # parser.add_argument('--save_path', type=str, required=False, default=None)
    args = parser.parse_args()

    X = np.load(args.data)    
    X = X.reshape(X.shape[0],-1,1)  # flatten spatial dimensions

    new = np.load(args.new)
    new = new.reshape(new.shape[0],-1,1)

    print(f'X shape: {X.shape}')
    print(f'new shape: {new.shape}')
    
    di_temp = []
    theta_temp = []
    st = time.time()
    for i in tqdm(range(new.shape[0])):
        temp = li_ar(
            X, new[i][None,...], ql=0.98, p=2, theta_fit="sueveges",
            p_value=None, dql=None, exp_test='anderson',
            p_cross=None, distributed='none',comm=None
        )
        
        di_temp.append(temp['d'])
        theta_temp.append(temp['theta'])
        # concat X and new[i] to X
        # X = np.concatenate((X, new[i][None,...]), axis=0) # we use same X (training data) for all new[i]
    total_time = time.time()-st
    time_per_sample = total_time/new.shape[0]

    di = np.array(di_temp)
    theta = np.array(theta_temp)

    save_path = os.path.dirname(args.new)
    np.save(os.path.join(save_path, 'd.npy'), di)
    np.save(os.path.join(save_path, 'theta.npy'), theta)
    
    print(f'Mean DI: {di.mean(axis=0)}')
    print(f'Mean theta: {theta.mean(axis=0)}')
    # print nans in theta
    print(f'Number of nans in theta: {np.isnan(theta).sum()}')

    print(f'elapsed time: {total_time} s')
    print(f'time per sample: {time_per_sample} s')
