#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:02:35 2021

@author: vinco
"""
import os
import sys
import time
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors



def embed_1dim(x, tau=1, m=2):

	# Check inputs shape
	if len(x.shape) != 1:
		if len(x.shape) == 2:
			Nx, Ny = x.shape
			if Nx == 1 and Ny == 1:
				raise ValueError(
					"The input x must be a time "
					"series with more than one value.")
			else:
				if Nx == 1:
					x = x[0,:]
				else:
					x = x[:,0]
		else:
			raise ValueError("The input x must be 1-dimensional.")
	if not isinstance(tau, int):
		raise ValueError("The input tau must be a scalar integer.")
	if not isinstance(m, int):
		raise ValueError("The input m must be a scalar integer.")

	# Now that x is 1-dim, extract number of epochs
	Nt = x.shape[0]
	# Create Hankel matrix
	H = np.zeros((m,Nt-(m-1)*tau))
	for mm in range(m):
		if mm==m-1:
			H[mm,:] = x[mm*tau:].T
		else:
			H[mm,:] = x[mm*tau:-((m-mm-1)*tau)].T
	return H



def _check_input_shape(X):
	if len(X.shape) == 1:
		Nt = X.shape[0]
		if Nt <= 1:
			raise ValueError(
				"The input X must have at least more than 1 observation.")
		else:
			X_tmp = X
			X = np.empty((Nt,1))
			X[:,0] = X_tmp.T
			del X_tmp
	elif len(X.shape) == 2:
		Nx, Ny = X.shape
		if Nx == 1 and Ny == 1:
			raise ValueError(
				"The input X must be a time series with more than one value.")
		elif Nx == 1 and Ny != 1:
			X = X.T
		X = X.reshape(X.shape + (1,))
	elif len(X.shape) == 3:
		n_samples, n_features, n_var = X.shape
	else:
		raise ValueError("The input X must be 1 or 2-dimensional.")
	return X



def embed(X, tau=[1], m=[2], t=None):
	# X: n-dim array of shape Nt, Nx where Nt is the number
	#    of observed epochs and Nx is the number of observables.
	# tau: list of integers containing the delay time for each
	#      of the observables (same length as Nx).
	# m: list of integers containing the embedding dimension
	#    for each of the observables (same length as Nx).

	# Check inputs shape
	if len(X.shape) == 1:
		Nt = X.shape[0]
		if Nt <= 1:
			raise ValueError(
				"The input X must have at least more than 1 observation.")
		else:
			X_tmp = X
			X = np.empty((Nt,1))
			X[:,0] = X_tmp.T
			del X_tmp
	elif len(X.shape) == 2:
		Nx, Ny = X.shape
		if Nx == 1 and Ny == 1:
			raise ValueError(
				"The input x must be a time series with more than one value.")
		elif Nx == 1 and Ny != 1:
			X = X.T
	else:
		raise ValueError("The input x must be 1 or 2-dimensional.")

	Nt, Nx = X.shape
	Ntau = len(tau)
	Nm = len(m)
	tauint = [int(tautau) for tautau in tau]
	mint = [int(mm) for mm in m]
	if tauint != tau:
		raise ValueError("The input tau must be a list of integers.")
	else:
		tau = np.array(tau)
	if mint != m:
		raise ValueError("The input m must be a list of integers.")
	else:
		m = np.array(m)
	if Ntau != Nx:
		raise ValueError(
			"The length of tau must equal the number of columns of X.")
	if Nm != Nx:
		raise ValueError(
			"The length of m must equal the number of columns of X.")

	NtH = Nt - np.max((m-1)*tau)
	if t is not None:
		tembed = t[-NtH:]
	else:
		tembed = None
	M = int(np.sum(m))
	H = np.zeros((M,NtH))
	kk = 0
	for ii in np.arange(Nx):
		Hii = embed_1dim(X[:,ii], tau=int(tau[ii]), m=int(m[ii]))
		NmHii, NtHii = Hii.shape
		H[kk:kk+m[ii],:] = Hii[:,NtHii-NtH:]
		kk += m[ii]
	return H.T, tembed




def _calc_tangent_map(
	X, t_step=1, n_neighbors=20, eps_over_L0=0.05,
	eps_over_L_fact=1.2, verbose=False):
	"""
	Calculate Lyapunov spectrum.

	Parameters
	----------
	X : TYPE
		Input data.
		Format: Nt x m, with Nt number of epochs,
		m number of time series.
	t_step : TYPE, optional
		DESCRIPTION. The default is 1.
	n_neighbors_min : TYPE, optional
		Minimum number of neighbors to use to calculate
		the tangent map. The default is 0, which uses
		m for n_neighbors_min.
	n_neighbors_max : TYPE, optional
		DESCRIPTION. The default is 20.
	eps_over_L0 : TYPE, optional
		Starting value for the distance to look up for
		neighbors, expressed as a fraction of the attractor
		size L. The default is 0.05.
	eps_over_L_fact : TYPE, optional
		Factor to increase the size of the neighborhood
		if not enough neighbors were found. The default
		is 1.2.

	Returns
	-------
	A : TYPE
		Tangent map approximation.
	eps_over_L : TYPE
		Final value for the distance to look up for neighbors,
		expressed as a fraction of the attractor size L, such
		that there are at least n_neighbors for the calculation
		of the tangent map at each epoch.
	"""
	# This function calculates an approximation
	# of the tangent map in the phase space.

	eps_over_L = eps_over_L0

	# Find number of epochs Nt and dimension m of X
	Nt, m, _ = X.shape

	# Find horizontal extent of the attractor
	L = np.max(X[:,-1]) - np.min(X[:,-1])
	flag_calc_map = True
	while flag_calc_map == True:

		# Set epsilon threshold
		eps = eps_over_L * L
		if verbose == True:
			print("eps_over_L = %f" %(eps_over_L))

		# Find first n_neighbors nearest neighbors
		# to each element X[tt,:]. The number of
		# neighbors is n_neighbors + 1 (because
		# the n_neighbors distances are calculated
		# also from the point itself and the distance
		# 0 needs to be esxcluded).
		nbrs = NearestNeighbors(
			n_neighbors=n_neighbors+1,
			algorithm='ball_tree').fit(X[:-t_step,:,0])

		# Find the distances and the indeces
		# of the nearest neighbors
		distances, indices = nbrs.kneighbors(X[:-t_step,:,0])

		# Find where the distances of the neighbours
		# are larger than the eps threshold
		ii = np.where(distances > eps)
		if len(ii[0]) > 0:
			eps_over_L = eps_over_L * eps_over_L_fact
		else:
			flag_calc_map = False

	# If n_neighbors_min is lower than the minimum
	# number of points required to estimate the
	# tangent map (i.e., lower than the dimension
	# of X), use as minimum number of neighbors
	# the minimum number necessary to calculate
	# the tangent map
	if n_neighbors < m: n_neighbors = m

	# Initialize the tangent map matrix A at each
	# epoch tt (if at time tt only n<n_neighbors
	# neighbors have a distance smaller than
	# eps, then retain only n neighbors).
	A = np.empty((Nt - t_step,m,m))

	# For every time step...
	for tt in np.arange(Nt - t_step):

		# The point under exam is X[tt,:]
		x0_nn = X[tt,:,0]

		# and it moves in X[tt+t_step,:] after t_step
		x0_nn1 = X[tt+t_step,:,0]

		# Create the variables containing the neighbors
		# at time tt (xneigh_nn) and their evolution
		# after t_step (xneigh_nn1)
		xneigh_nn  = X[indices[tt],:,0]
		xneigh_nn1 = X[indices[tt]+t_step,:,0]

		# Calculate the distances of the neighbors
		# from the point under exam (exclude the
		# first element of xneigh_nn because it
		# is equal to x0_nn)
		y = xneigh_nn[1:] - x0_nn

		# Calculate the distances of the neighbors' evolution from
		# the evolution in time of the point under exam (exclude the
		# first element of xneigh_nn1 because it is equal to x0_nn1)
		z = xneigh_nn1[1:] - x0_nn1

		# Calculate the tangent map A at time
		# tt using the pseudo-inverse of y.T
		A[tt,:,:] = np.dot(z.T,np.linalg.pinv(y.T))

	# Return the tangent map A
	return A, eps_over_L


def calc_lyap_spectrum_serial(
	X, dt=1, t_step=1, n_neighbors=20, eps_over_L0=0.05, eps_over_L_fact=1.2,
	sampling=['rand',100], n=1000, method="SS85", verbose=False,
	flag_calc_tangent_map=True, A=None):
	# sampling: to decide which points to use for
	#           the Lyapunov spectrum estimation.
	# 			Options:
	#    			['all', None]: Use all the possible trajectories.
	#    			['begin', int]: Start from the beginning of the
	#    			                time series and take a new trajectory
	# 								after int steps.
	#    			['mid', int]: Start from the middle of the time
	#    			              series and take a new trajectory
	# 							  after int steps.
	#    			['rand', None]: Start from allowed random times.

	Nt, m, _ = X.shape

	if method == "SS85":
		# Find horizontal extent of the attractor
		L = np.max(X[:,-1]) - np.min(X[:,-1])
		if flag_calc_tangent_map == True:
			if verbose == True:
				tic = time.time()
				print("")
				print("Calculating tangent map: ", end='')

			A, eps_over_L = _calc_tangent_map(
				X, t_step=t_step,
				n_neighbors=n_neighbors,
				eps_over_L0=eps_over_L0,
				eps_over_L_fact=eps_over_L_fact
			)

			if verbose == True:
				print(
					"eps_over_L = %f   %.2f s" %(
						eps_over_L, time.time() - tic))
		else:
			if A == None:
				raise ValueError('Tangent map is missing.')

		nbrs = NearestNeighbors(
			n_neighbors = n_neighbors + 1,
			algorithm='ball_tree'
		).fit(X[:-t_step,:,0])
		distances, indices = nbrs.kneighbors(X[:,:,0])

		if sampling[0] == 'all':
			if sampling[1] == None:
				ts = np.arange(0, Nt-n*t_step, 1)
			else:
				raise ValueError(
					'When sampling[0] is ''all'', sampling[1] must be None')
		elif sampling[0] == 'mid':
			ts = np.arange(int(Nt/2), Nt - n * t_step,sampling[1])
		elif sampling[0] == 'begin':
			ts = np.arange(0,Nt-n*t_step,sampling[1])
		elif sampling[0] == 'rand':
			Nles_statistic = sampling[1]
			ts = np.sort(np.array([int(ii) for ii in \
					np.floor(np.random.rand(Nles_statistic) * \
						(Nt-n*t_step))])
					)
		else:
			raise ValueError('sampling[0] not valid.')

		Nles_statistic = len(ts)
		logR = np.zeros((Nles_statistic,n,m))
		les = np.empty((Nles_statistic,n,m))
		kk = -1
		if verbose == True:
			print("")
			print("Calculating Lyapunov spectrum")
		les_mean = np.zeros((n,m))
		les_std  = np.zeros((n,m))
		for t0 in tqdm(ts):
			kk += 1
			ind2follow = indices[t0,:]
			distances_ind2follow = distances[t0,:]
			ind2rm = np.where(ind2follow+(t0+n*t_step)>Nt)[0]
			ind2follow = np.delete(ind2follow, ind2rm)
			distances_ind2follow = np.delete(distances_ind2follow, ind2rm)
			jj2rm = distances_ind2follow>(eps_over_L * L)
			ind2follow = np.delete(ind2follow, jj2rm)
			ind2follow = ind2follow[1:]
			e = np.eye(m)

			for nn in np.arange(n):
				ii = t0 + nn * t_step
				Aii = A[ii,:,:]
				NA = 0
				NR = 0
				if np.sum(np.abs(Aii)) == 0.0:
					logR[kk,nn:,:] = np.nan
					les[kk,nn:,:] = np.nan
					if NA != 0:
						NA = nn
				else:
					Ae = np.dot(Aii,e)
					Q, R = np.linalg.qr(Ae)
					if nn > 0:
						logR[kk,nn,:] = np.log(np.abs(np.diag(R)))
						if np.abs(np.sum(logR[kk,nn,:])) == np.inf:
							les[kk,nn,:] = np.nan
							if NR != 0: NR = nn
						else:
							les[kk,nn,:] = (1 / (nn * t_step * dt)) * \
								np.sum(logR[kk,1:nn,:], axis=0)
					e = Q
				if NA != 0: les[kk,NA:,:] = np.nan
				if NR != 0: les[kk,NR:,:] = np.nan

				# Temporary ignore RunTimeWarning when performing
				# np.nanmean and np.nanstd on arrays containing only nan
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					les_mean[nn,:] = np.nanmean(les[:,nn,:], axis=0)
					les_std [nn,:] = np.nanstd (les[:,nn,:], axis=0)
		if verbose==True:
			print("\nles: ", end='')
			print(les_mean[-1,:])
			print("les std: ", end='')
			print(les_std[-1,:])

	else:
		raise ValueError("Select a valid method")
	return les, les_mean, les_std, eps_over_L



def calc_lyap_spectrum_parallel_traj(
	X, dt=1, t_step=1, n_neighbors=20, eps_over_L0=0.05, eps_over_L_fact=1.2,
	sampling=['rand',100], n=1000, method="SS85", verbose=False, comm=None):
	"""
	sampling: to decide which points to use for
	          the Lyapunov spectrum estimation.
				Options:
	   			['all', None]: Use all the possible trajectories.
	   			['begin', int]: Start from the beginning of the
	   			                time series and take a new trajectory
									after int steps.
	   			['mid', int]: Start from the middle of the time
	   			              series and take a new trajectory
								  after int steps.
	   			['rand', None]: Start from allowed random times.
	"""

	n_samples, n_features, _ = X.shape

	if comm == None:
		rank = 0
		size = 1
	else:
		from mpi4py import MPI
		rank = comm.Get_rank()
		size = comm.Get_size()
		comm.Barrier()

	if method == "SS85":
		### start comment
		### Could make all these calculations only on rank 0 and then share

		# Find horizontal extent of the attractor
		L = np.max(X[:,-1]) - np.min(X[:,-1])

		# define ts, i.e. the indices of the trajectories to be followed
		if sampling[0] == 'all':
			if sampling[1] == None:
				ts = np.arange(0, n_samples-n*t_step, 1)
			else:
				raise ValueError(
					'When sampling[0] is ''all'', sampling[1] must be None')
		elif sampling[0] == 'mid':
			ts = np.arange(int(n_samples/2), n_samples - n * t_step,sampling[1])
		elif sampling[0] == 'begin':
			ts = np.arange(0,n_samples-n*t_step,sampling[1])
		elif sampling[0] == 'rand':
			Nles_statistic = sampling[1]
			ts = np.sort(np.array([int(ii) for ii in \
					np.floor(np.random.rand(Nles_statistic) * \
						(n_samples-n*t_step))])
					)
		else:
			raise ValueError('sampling[0] not valid.')

		# define how many trajectories will be followed
		Nles_statistic = len(ts)

		# find distances and indices of the nearest neighbors for each
		# time series
		nbrs = NearestNeighbors(
			n_neighbors = n_neighbors + 1,
			algorithm='ball_tree'
		).fit(X[:-t_step,:,0])
		distances, indices = nbrs.kneighbors(X[:,:,0])

		flag_calc_map = True
		eps_over_L = eps_over_L0
		while flag_calc_map == True:
			# Set epsilon threshold
			eps = eps_over_L * L
			# Find where the distances of the neighbours
			# are larger than the eps threshold
			jj = np.where(distances[:-t_step] > eps)
			if len(jj[0]) > 0:
				eps_over_L = eps_over_L * eps_over_L_fact
			else:
				flag_calc_map = False

		# if n_neighbors_min is lower than the minimum
		# number of points required to estimate the
		# tangent map (i.e., lower than the dimension
		# of X), use as minimum number of neighbors
		# the minimum number necessary to calculate
		# the tangent map
		if n_neighbors < n_features: n_neighbors = n_features

		### Could make all these calculations only on rank 0 and then share
		### end comment

		perrank = Nles_statistic // size
		remaind = Nles_statistic % size
		if rank == size - 1:
			n_sample_rank = perrank + remaind
		else:
			n_sample_rank = perrank

		# initialize variables
		logR = np.zeros((n_sample_rank,n,n_features))
		les = np.empty((n_sample_rank,n,n_features))
		comm.Barrier()
		if rank != size - 1:
			# for every trajectory
			#for t0 in tqdm(ts):
			for kk,t0 in enumerate(ts[rank * perrank : (rank + 1) * perrank]):
				les = time_parallel_loop(
					X=X, t0=t0, indices=indices, distances=distances,
					n=n, t_step=t_step, dt=dt, n_samples=n_samples,
					n_features=n_features, eps_over_L=eps_over_L, L=L,
					les=les, logR=logR, kk=kk
				)
		else:
			## loop over remaining times (remainder)
			pbar = tqdm(total = n_sample_rank, desc='# iterations on epochs')
			for kk, t0 in enumerate(ts[(size-1) * perrank : Nles_statistic]):
				pbar.update(1)
				les = time_parallel_loop(
					X=X, t0=t0, indices=indices, distances=distances, n=n,
					t_step=t_step, dt=dt, n_samples=n_samples,
					n_features=n_features, eps_over_L=eps_over_L, L=L, les=les,
					logR=logR, kk=kk
				)

		comm.Barrier()
		les = comm.gather(les, root=0)
		## rank 0 operations
		if rank == 0:
			## concatenated vectors after gathering
			les  = np.concatenate(les, axis=0)
			les_mean = np.zeros((n,n_features))
			les_std  = np.zeros((n,n_features))
			for nn in np.arange(n):
				# temporary ignore RunTimeWarning when performing
				# np.nanmean and np.nanstd on arrays containing only nan
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					les_mean[nn,:] = np.nanmean(les[:,nn,:], axis=0)
					les_std [nn,:] = np.nanstd (les[:,nn,:], axis=0)

			if verbose==True:
				print("\nles: ", end='')
				print(les_mean[-1,:])
				print("les std: ", end='')
				print(les_std[-1,:])
		else:
			les = None
			les_mean = None
			les_std = None
			eps_over_L = None

	else:
		raise ValueError("Select a valid method")
	return les, les_mean, les_std, eps_over_L, rank



def time_parallel_loop(
	X, t0, indices, distances, n, t_step, dt, n_samples, n_features,
	eps_over_L, L, les, logR, kk):

	# extract the index to follow
	ind2follow = indices[t0,:]
	# find the distances associated with the trajectory to follow
	distances_ind2follow = distances[t0,:]
	# if the trajectory to be followed ends before the number of
	# steps that we want to follow, mark that trajectory as to
	# be removed
	ind2rm = np.where(ind2follow+(t0+n*t_step)>n_samples)[0]
	# remove the indices and distances associated with trajectories
	# listed as to be removed
	ind2follow = np.delete(ind2follow, ind2rm)
	distances_ind2follow = np.delete(distances_ind2follow, ind2rm)
	# remove trajectories that are too distant
	jj2rm = distances_ind2follow>(eps_over_L * L)
	ind2follow = np.delete(ind2follow, jj2rm)
	# remove closest trajectory because it is the point if interest
	# itself but we are interested in the neighbors
	ind2follow = ind2follow[1:]

	# create orthonormal base in dimension n_features
	e = np.eye(n_features)
	# follow the trajectories for n steps
	# for each of these steps do
	for nn in np.arange(n):
		# calculate time stamp
		ii = t0 + nn * t_step

		# estimate tangent map at the given time stamp
		# this is a n_features x n_features matrix;
		# the point under exam is X[ii,:]
		x0_nn = X[ii,:,0]

		# and it moves in X[ii+t_step,:] after t_step
		x0_nn1 = X[ii+t_step,:,0]

		# create the variables containing the neighbors
		# at time tt (xneigh_nn) and their evolution
		# after t_step (xneigh_nn1)
		xneigh_nn  = X[indices[ii],:,0]
		xneigh_nn1 = X[indices[ii]+t_step,:,0]

		# calculate the distances of the neighbors
		# from the point under exam (exclude the
		# first element of xneigh_nn because it
		# is equal to x0_nn)
		y = xneigh_nn[1:] - x0_nn

		# calculate the distances of the neighbors' evolution from
		# the evolution in time of the point under exam (exclude the
		# first element of xneigh_nn1 because it is equal to x0_nn1)
		z = xneigh_nn1[1:] - x0_nn1

		# calculate the tangent map A at time
		# tt using the pseudo-inverse of y.T
		Aii = np.dot(z.T,np.linalg.pinv(y.T))

		# set control values to see if either the tangent map is
		# null
		NA = 0
		# or an element of the diagonal of R in the QR-decomposition
		# is null
		NR = 0
		# if the tangent map is null set all variables to nan
		if np.sum(np.abs(Aii)) == 0.0:
			logR[kk,nn:,:] = np.nan
			les[kk,nn:,:] = np.nan
			if NA != 0:
				NA = nn
		# if the tangent map is not null
		else:
			# dot product of the orthonormal base e and the
			# tangent map
			Ae = np.dot(Aii,e)
			# qr-factorization of Ae, i.e. Ae = Q*R where
			# Q contains the base
			# the diagonal of R contains the products <A_j, e_j>
			Q, R = np.linalg.qr(Ae)
			# if the time step is not the first
			if nn > 0:
				# the quantity of interest is the log of the abs
				# of <A_j, e_j>
				# (see eq. 10 of Sano and Sawada, 1985, PRL)
				logR[kk,nn,:] = np.log(np.abs(np.diag(R)))
				# if there is a null value in diag(R)
				if np.abs(np.sum(logR[kk,nn,:])) == np.inf:
					# set results to nan
					les[kk,nn,:] = np.nan
					# and set NR to the time step nn
					if NR != 0: NR = nn
				else:
					# calculate the Lyapunov exponents for the kk-th
					# trajectory out of the Nles_statistic at time
					# step nn out of the n
					les[kk,nn,:] = (1 / (nn * t_step * dt)) * \
						np.sum(logR[kk,1:nn,:], axis=0)
			# reset the orthonormal base to the orthonormal matrix Q
			e = Q
		# if we got a null tangent map set the Lyapunov exponents to
		# nan from the time step NA (i.e., the one where the tangent
		# map was null)
		if NA != 0: les[kk,NA:,:] = np.nan
		# if we got a null diag(R) element set the Lyapunov
		# exponents to nan from the time step NR (i.e., the one
		# where we found a null diagonal element in R)
		if NR != 0: les[kk,NR:,:] = np.nan
	return les


def compute(
	X, dt=1, t_step=1, n_neighbors=20, eps_over_L0=0.05,
	eps_over_L_fact=1.2, sampling=['rand',100], n=1000, method="SS85",
	verbose=False, flag_calc_tangent_map=True, A=None, distributed='none',
	comm=None):

	X = _check_input_shape(X)
	if distributed == 'none':
		rank = 0
		_, les_mean, les_std, _ = calc_lyap_spectrum_serial(
			X=X,
			dt=dt,
			t_step=t_step,
			n_neighbors=n_neighbors,
			eps_over_L0=eps_over_L0,
			eps_over_L_fact=eps_over_L_fact,
			sampling=sampling,
			n=n,
			method=method,
			verbose=verbose,
			flag_calc_tangent_map=flag_calc_tangent_map,
			A=A
		)
	elif distributed == 'traj':
		_, les_mean, les_std, _, rank = calc_lyap_spectrum_parallel_traj(
			X=X,
			dt=dt,
			t_step=t_step,
			n_neighbors=n_neighbors,
			eps_over_L0=eps_over_L0,
			eps_over_L_fact=eps_over_L_fact,
			sampling=sampling,
			n=n,
			method=method,
			verbose=verbose,
			comm=comm)

 	## get the results for rank 0 only
	if rank == 0:
		les_mean_last = les_mean[-1,:]
		les_std_last = les_std[-1,:]
		H_last = np.sum(les_mean_last[les_mean_last>0])
		results = {
			'les_mean': les_mean_last,
			'les_std' : les_std_last,
			'H'       : H_last,
		}
		return results
