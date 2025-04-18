import numpy as np
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='data/lorenz/lorenz.npy')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    args = parser.parse_args()

    data_root = os.path.dirname(args.data)
    os.makedirs(os.path.join(data_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'val'), exist_ok=True)
    os.makedirs(os.path.join(data_root, 'test'), exist_ok=True)

    data = np.load(args.data).squeeze()
    print(f"Loaded data {data.shape}")
    n_samples = data.shape[0]
    n_train = int(n_samples * args.train_ratio)
    n_val = int(n_samples * args.val_ratio)
    n_test = n_samples - n_train - n_val

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # save mean and std of train to data_root
    train_mean = train_data.mean(axis = 0)  #time axis
    train_std = train_data.std(axis = 0)
    np.save(os.path.join(data_root, 'mean.npy'), train_mean)
    np.save(os.path.join(data_root, 'std.npy'), train_std)

    #normalize
    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    np.save(os.path.join(data_root, 'train', 'data.npy'), train_data)
    np.save(os.path.join(data_root, 'val', 'data.npy'), val_data)
    np.save(os.path.join(data_root, 'test', 'data.npy'), test_data)
    
    print(f"Save data to {data_root}")
