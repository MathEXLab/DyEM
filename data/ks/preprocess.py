# calculate vorcity
import numpy as np
import os
import time
import h5py
from argparse import ArgumentParser


    
def read_h5py(filename, varname = 'velocity_field'):
    
    data_path = filename
    with h5py.File(data_path, 'r') as f:
        data = f[varname][:]
    
    print('data shape:', data.shape)
    return data

# truncate then downsample
def downsample_and_truncate(data, interval = 10, truncate_before = 30000):
    print('data shape before downsample and truncate:', data.shape)
    data = data[truncate_before:,...]
    data = data[::interval,...]
    print('data shape after downsample and truncate:', data.shape)
    return data

def calc_vorticity(data):
    print('calculating vorcity...')
    start = time.time()

    u = data[...,0]
    v= data[...,1]
    q = np.gradient(u, axis=1) - np.gradient(v, axis=0)
    print('q.shape = ', q.shape)
    print('elapsed time = ', time.time() - start)

    return q

def split_data(data, train_ratio = 0.7, val_ratio = 0.15, save_dir = 'data', save = False):

    os.makedirs(os.path.join(save_dir,'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir,'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir,'test'), exist_ok=True)

    train = data[:int(data.shape[0]*train_ratio),...]
    val = data[int(data.shape[0]*train_ratio):int(data.shape[0]*(train_ratio+val_ratio)),...]
    test = data[int(data.shape[0]*(train_ratio+val_ratio)):,...]
    print('train.shape = ', train.shape)
    print('val.shape = ', val.shape)
    print('test.shape = ', test.shape)

    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)

    # standardize
    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std

    # save to npy
    if save:
        np.save(os.path.join(save_dir,'train','data.npy'), train)
        np.save(os.path.join(save_dir,'val','data.npy'), val)
        np.save(os.path.join(save_dir,'test','data.npy'), test)
        np.save(os.path.join(save_dir,'mean.npy'), mean)
        np.save(os.path.join(save_dir,'std.npy'), std)


    return train, val, test
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default = '/home/dynamical_embedding/data/kolmogorov/nf2/Re_14.4_nf2.h5')
    parser.add_argument('--interval', type=int, default=50) #dt = 0.5
    parser.add_argument('--truncate_before', type=int, default=30000)
    parser.add_argument('--save_dir', type=str, default='/home/dynamical_embedding/data/kolmogorov')
    parser.add_argument('--save', type=bool, default=True)
    args = parser.parse_args()

    data = read_h5py(args.data_path)
    print('data loaded')
    data = downsample_and_truncate(data, args.interval, args.truncate_before)
    q = calc_vorticity(data)
    train, val, test = split_data(q, save_dir = args.save_dir, save = args.save)
    print(f'data saved to {args.save_dir}')