# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from KS import KS


############################################################################
# set public parameters here
############################################################################
# L   = 1/np.sqrt(0.085)           # domain is 0 to 2.*np.pi*L
L = 3.50
N   = 64          # number of collocation points
dt  = 0.01          # time step -- For same setting with ref, use 0.01
diffusion = 1.0
############################################################################


def ks_generator(length,name):
    # define a KS model
    ks = KS(L=L,diffusion=diffusion,N=N,dt=dt)
    # random initial condition
    #u = np.cos(x/L)*(1.0+np.sin(x/L)) # smooth IC
    u = 0.01*np.random.normal(size=N) # noisy IC
    # remove zonal mean
    u = u - u.mean()
    # spectral space variable.
    ks.xspec[0] = np.fft.rfft(u)

    # generate data
    us = []; ts = []
    st = time.time()
    for i in range(length):
        ks.advance()
        u = ks.x.squeeze()
        us.append(u); ts.append(i*dt)
    print(f'data generated, time:{time.time()-st}') 

    # save data
    save_dir = '/home/dynamical_embedding/data/ks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir,name)):
        os.makedirs(os.path.join(save_dir,name))
    np.save(os.path.join(save_dir,name,'u.npy'), us)
    np.save(os.path.join(save_dir,name,'t.npy'), ts)

    return os.path.join(save_dir,name)

def LPfilter(path, rank = 16, save = False):
    u = np.load(os.path.join(path,'u.npy'))
    uu = np.multiply(u,u)
    t = np.load(os.path.join(path,'t.npy'))

    # filter u
    u_f = np.fft.rfft(u, axis=0) # fft
    freq = np.fft.rfftfreq(u.shape[0], d = t[1]-t[0])
    lpfilter = np.concatenate((np.ones((rank,1)), np.zeros((u_f.shape[0]-rank,1)))) # low-pass filter
    u_filtered = np.multiply(u_f,lpfilter) 
    u_bar = np.fft.irfft(u_filtered, axis = 0) # ifft

    # filter uu
    uu_f = np.fft.rfft(uu, axis=0)
    uu_filtered = np.multiply(uu_f,lpfilter)
    uu_bar = np.fft.irfft(uu_filtered, axis = 0)

    # calculate stress
    stress = get_stress(u_bar, uu_bar)

    # calculate strain
    strain = get_strain(u_bar)

    # save
    if save:
        np.save(os.path.join(path,'u_bar.npy'),u_bar) # currently don't need intermiate results
        np.save(os.path.join(path,'uu_bar.npy'),uu_bar)
        np.save(os.path.join(path,'stress.npy'),stress)
        np.save(os.path.join(path,'strain.npy'),strain)

# def GaussianFilter(path, a = 1, save = False):
#     '''
#     The Dirac delta function is approximated by Gaussian function
#     '''
#     # a - smaller a means more similar to delta function
#     u = np.load(os.path.join(path,'u.npy'))
#     uu = np.multiply(u,u)
#     t = np.load(os.path.join(path,'t.npy'))

#     # filter u
#     u_f = np.fft.rfft(u, axis=0) # fft
#     freq = np.fft.rfftfreq(u.shape[0], d = t[1]-t[0])
#     x = np.linspace(-2,2,u_f.shape[0])
#     gaussian_filter = 1/abs(a)/np.sqrt(np.pi)*np.exp(-(x/a*x/a)) # delta_filter use Gaussian form
#     gaussian_filter = np.fft.rfft(gaussian_filter, axis=0) # fft
#     delta_filter = np.expand_dims(delta_filter,axis = 1)
#     # lpfilter = np.concatenate((np.ones((rank,1)), np.zeros((u_f.shape[0]-rank,1)))) # low-pass filter
#     u_filtered = np.multiply(u_f,delta_filter) 
#     u_bar = np.fft.irfft(u_filtered, axis = 0) # ifft

#     # filter uu
#     uu_f = np.fft.rfft(uu, axis=0)
#     uu_filtered = np.multiply(uu_f,delta_filter)
#     uu_bar = np.fft.irfft(uu_filtered, axis = 0)

#     # calculate stress
#     stress = get_stress(u_bar, uu_bar)

#     # calculate strain
#     strain = get_strain(u_bar)

#     # save
#     if save:
#         np.save(os.path.join(path,'u_bar.npy'),u_bar) # currently don't need intermiate results
#         np.save(os.path.join(path,'uu_bar.npy'),uu_bar)
#         np.save(os.path.join(path,'stress.npy'),stress)
#         np.save(os.path.join(path,'strain.npy'),strain)

def AllPassfilter(path, save = False):
    u = np.load(os.path.join(path,'u.npy'))
    uu = np.multiply(u,u)
    t = np.load(os.path.join(path,'t.npy'))

    # filter u
    u_f = np.fft.rfft(u, axis=0) # fft
    freq = np.fft.rfftfreq(u.shape[0], d = t[1]-t[0])
    apfilter = np.ones((u_f.shape[0],1)) # low-pass filter
    u_filtered = np.multiply(u_f,apfilter) 
    u_bar = np.fft.irfft(u_filtered, axis = 0) # ifft

    # filter uu
    uu_f = np.fft.rfft(uu, axis=0)
    uu_filtered = np.multiply(uu_f,apfilter)
    uu_bar = np.fft.irfft(uu_filtered, axis = 0)

    # calculate stress
    stress = get_stress(u_bar, uu_bar)

    # calculate strain
    strain = get_strain(u_bar)

    # save
    if save:
        np.save(os.path.join(path,'u_bar.npy'),u_bar) # currently don't need intermiate results
        np.save(os.path.join(path,'uu_bar.npy'),uu_bar)
        np.save(os.path.join(path,'stress.npy'),stress)
        np.save(os.path.join(path,'strain.npy'),strain)

def SpaceLowWave(path, rank = 16, save = False):
    u = np.load(os.path.join(path,'u.npy'))
    uu = np.multiply(u,u)
    t = np.load(os.path.join(path,'t.npy'))

    # filter u
    u_f = np.fft.rfft(u, axis=1) # spactial fft
    # freq = np.fft.rfftfreq(u.shape[0], d = t[1]-t[0])
    apfilter = np.concatenate((np.ones((rank,1)), np.zeros((u_f.shape[1]-rank,1))), axis = 0) # spatial filter
    u_filtered = np.multiply(u_f,apfilter.T) 
    u_bar = np.fft.irfft(u_filtered, axis = 1) # ifft

    # filter uu
    uu_f = np.fft.rfft(uu, axis=1)
    uu_filtered = np.multiply(uu_f,apfilter.T)
    uu_bar = np.fft.irfft(uu_filtered, axis = 1)

    # calculate stress
    stress = get_stress(u_bar, uu_bar)

    # calculate strain
    strain = get_strain(u_bar)

    # save
    if save:
        np.save(os.path.join(path,'u_bar.npy'),u_bar) # currently don't need intermiate results
        np.save(os.path.join(path,'uu_bar.npy'),uu_bar)
        np.save(os.path.join(path,'stress.npy'),stress)
        np.save(os.path.join(path,'strain.npy'),strain)

def get_stress(u_bar, uu_bar):
    tao = uu_bar - np.multiply(u_bar,u_bar)
    return tao

def get_strain(u_bar):
    strain = np.zeros(u_bar.shape)
    dx = 2*np.pi/np.sqrt(0.085)/256
    for i in range(256):
        if i != 0 and i !=255:
            strain[:,i] = (u_bar[:,i+1] - u_bar[:,i-1])/2*dx
        elif i == 0:
            strain[:,i] = (u_bar[:,i+1] - u_bar[:,i])/dx
        elif i == 255:
            strain[:,i] = (u_bar[:,i] - u_bar[:,i-1])/dx
            
    return strain
    
if __name__ == '__main__':
    # generate ke data
    filter_name = 'AllPass' # AllPass, LowPass, Gaussian
    save_dir = ks_generator(length = 2500000, name = filter_name) # name is used for creating a subfolder

    # apply filter -- all filter to choose
    # AllPassfilter(save_dir,save = True)
    # LPfilter(save_dir, rank = 16) 
    # SpaceLowWave(save_dir, rank = 16, save = True)

    
