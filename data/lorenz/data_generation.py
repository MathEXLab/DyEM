import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
from argparse import ArgumentParser

class Lorenz:
    def __init__(self,
    
                 n_samples=10000,
                 nt = 100,
                 sample_space = 20.0):
        '''
        s, r, b : float
            Parameters defining the Lorenz attractor.
            s=10, r=28, b=2.667： Lorenz63
        sample_space: float
            initial condition of x, y,  z will be sampled in this range, e.g. (-20,+20)
        n_samples: int
            number of samples to generate
        '''
        self.s = s
        self.r = r
        self.b = b
        self.n_samples = n_samples
        self.nt = nt
        self.sample_space = sample_space
        self.dt = 0.01  # fix to avoid discrete problem
        print('----------')
        print('data summary:')
        print('----------')
        print(f's:{s}')
        print(f'r:{r}')
        print(f'b:{b}')
        print(f'n_samples:{n_samples}')
        print(f'nt:{nt}')
        print(f'sample_space:{sample_space}')
        print('----------')

    def gen_state(self, x0):
        """
        Parameters
        ----------
        xyz : array-like, shape (3,)

        Returns
        -------
        xyz_dot : array, shape (3,)
        Values of the Lorenz attractor's partial derivatives at *xyz*.
        """
        x, y, z = x0
        x_dot = self.s*(y - x)
        y_dot = self.r*x - y - x*z
        z_dot = x*y - self.b*z
        return np.array([x_dot, y_dot, z_dot])

    def gen_series(self, save_path=None):
        self.xyzs = np.empty((self.n_samples, self.nt + 1, 3))  # Need one more for the initial values
        self.xyzs[:,0,:] = ((np.random.rand(self.n_samples, 3)-0.5)*2*self.sample_space)  # initialization space [-sample_space, sample_space]
        for i_sample in range(self.n_samples):
            for i in range(self.nt):
                self.xyzs[i_sample, i + 1, :] = self.xyzs[i_sample, i, :] + self.gen_state(self.xyzs[i_sample, i, :]) * self.dt

        if save_path is not None:
            np.save(save_path, self.xyzs)
            print(f'save data to {save_path}')

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--s', type=float, default=10)
    parser.add_argument('--r', type=float, default=28)
    parser.add_argument('--b', type=float, default=2.667)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--nt', type=int, default=10000)
    parser.add_argument('--sample_space', type=float, default=20.0)
    parser.add_argument('--save_path', type=str, default='data')
    args = parser.parse_args()

    sys = Lorenz(s=args.s, r=args.r, b=args.b, n_samples=args.n_samples, nt=args.nt,sample_space=args.sample_space)
    os.makedirs(os.path.join(args.save_path, 'lorenz'), exist_ok=True)
    save_path = os.path.join(args.save_path, 'lorenz', 'lorenz.npy')
    sys.gen_series(save_path = save_path)
