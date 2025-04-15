import os
import sys
import time
import pickle
import numpy as np


def check_inputs(X):
	"""
	Checks that data format has correct shape,
	and assigns consistent axis for dask operations
	(inconsistent with numpy).
	"""
	## make sure it is a 3D matrix
	if X.ndim == 2:
		X = X.reshape(X.shape + (1,))
	if X.ndim != 3:
		raise ValueError('Input matrix `X` must be 3D.')

	## for dask operation (axis not consistent with numpy)
	if isinstance(X, xr.DataArray):
		axis = 0
	else:
		axis = 1
	return X, axis


def _embed_1d(x, m, tau):
	"""
	Embeds a one dimensional column vector.
	x:
	m:
	tau:
	"""
	## check dimensions x
	x = np.atleast_2d(x)
	if x.shape[1] != 1:
		raise ValueError('`x` must be a column vector of dimension n x 1')
	## compute one dimensional embedding
	n_samples = x.shape[0]
	n_samples_e = n_samples - (m - 1) * tau
	e = np.zeros([n_samples_e, m])
	for jj in range(m):
		if jj == m - 1:
			e[:,jj] = x[jj*tau:]
		else:
			e[:,jj] = x[jj*tau:-(m-jj-1)*tau]
	return e


def embed_nd(X, tau=[1], m=[2], t=None, centering=None):
	"""
	X: input data of shape [n_samples, n_features, n_variables].
	tau: embedding delay time
	m: embedding dimension
	t: data time steps (must be uniform)
	centering: type of embedding matrix centering.
				Options: `None` (default), `demean`

	Returns:
	embedding: embedding matrix
	t_embedding: time associated with the embedding matrix
	"""
	## check input data
	X = check_inputs(X)

	## input data shape
	n_samples, n_features, n_variables = X.shape

	## check type and length of tau and m
	if (len(tau) != 1) and (len(tau) != n_variables):
		raise ValueError('len(tau) must be = 1 or = n_variables')
	if (len(m) != 1) and (len(m) != n_variables):
		raise ValueError('len(m) must be = 1 or = n_variables')
	tau = [int(tau) for i in tau]
	m = [int(m) for i in m]

	# expand list to match n_variables
	if len(tau) == 1: tau = tau * n_variables
	if len(m)   == 1: m   = m   * n_variables

	## set dimensions embedding matrix
	var_max = np.argmax([(j - 1) * i for i,j in zip(tau,m)])
	n_samples_embedding = n_samples - (m[var_max] - 1) * tau[var_max]
	n_variables_embedding = int(np.sum(m))

	## time embedding assignment and initialization embedding matrix
	if t is not None: t_embedding = t[-n_samples_embedding:]
	else: t_embedding = None
	embedding = np.zeros([
		n_samples_embedding, n_features, n_variables_embedding])

	m0 = 0
	for n_feature in range(n_features):
		for n_variable in range(n_variables):
			n_sample = n_samples - (m[n_variable]-1)*tau[n_variable]
			embedding[-n_sample:,n_feature,m0:m0+m[n_variable]] = embed_1d(
				X[:,n_features, n_variable],
				tau=tau[n_variable],
				m=m[n_variable]
			)
			m0 += m[n_variable]
	return embedding, t_embedding


def _best_embedding_params_univ(
		x, m_range=range(1,20), tau_range=range(1,1000), method_tau='mi',
		method_m='cao', bins=64, metric='minkowski', p=2, theiler=None,
		n_reps_autocorr_time=100, E1_thresh=0.95, E2_thresh=0.95):
	"""
	theiler: minimum temporal distance between reference point and a neighbor
	"""
	n_taus = len(tau_range)
	tau_max = np.max(tau_range)
	n_samples = x.shape[0]

	## Find first minimum of the mutual information between the time series
	## and its lagged copies
	if method_tau == 'mi':
		X = np.empty([n_samples-n_taus, n_taus])
		## reference time series
		x1 = x[:-tau_max-1]
		## initialize mutual information
		mi = np.empty([n_taus,])

		for idx,tau in enumerate(tau_range):
			## time lagged time series
			x2 = x[tau:-tau_max+tau-1]
			## joint pdf
			c_x1x2 = np.histogram2d(x1, x2, bins)[0]
			## mutual information calculation
			mi[idx] = mis(None, None, contingency=c_x1x2)
		## first mi minimum
		idx_tau_opt = np.argmax(np.diff(mi)>0)
		## if no negative diff (i.e., mi always decreasing), np.argmax
		## gives 0 as output, but we want the last element in this case
		## (global minimum)
		if idx_tau_opt == 0:
			idx_tau_opt = -1
		## set tau_opt from tau_range
		tau_opt = tau_range[idx_tau_opt]
	if method_m == 'cao':
		tic = time.time()
		if theiler is not None:
			W0 -= 1
		else:
			autocorr_time, _ = _calc_autocorr_time(
				x.reshape(-1,1), n_reps=n_reps_autocorr_time)
		idx_m1 = -1
		E = np.empty([len(m_range)-1,])
		Estar = np.empty([len(m_range)-1,])
		## m1 and m2 are the tested embedding dimensions (with m1<m2)
		for m2,m1 in zip(m_range[1:],m_range[:-1]):
			idx_m1 += 1
			## embed x with m2 and tau_opt
			X2 = _embed(x,m=m2,tau=tau_opt)
			n_samples_embed = X2.shape[0]
			## embed x with m1 and tau_opt (keep only points at same time of X2)
			X1 = _embed(x,m=m1,tau=tau_opt)[:n_samples_embed,:]

			if theiler is None:
				W0 = int(autocorr_time*((2/n_samples_embed)**(2/m1)))
			flag_theiler = True
			W = W0
			while flag_theiler == True:
				W += 1
				k = 2 * W + 3
				knn1 = neighbors.KNeighborsRegressor(k+1, p=p, metric=metric)
				## keep only the indices that are less than library length
				knn1.fit(X1, X1)
				dist1,ind1 = knn1.kneighbors(X1)
				dist_bool = dist1>0.0
				dist_bool_nn = np.zeros_like(dist_bool, dtype=bool)
				ind_nn = dist_bool.argmax(axis=1)
				idx = np.arange(n_samples_embed), ind_nn
				dist_bool_nn[idx] = dist_bool[idx]
				if len(ind_nn[ind_nn==0]) == 0:
					dist1_nn = dist1[dist_bool_nn]
					ind1_nn = ind1[dist_bool_nn]
					flag_theiler = False
			dist2_nn = (np.sum((X2[ind1_nn,:]-X2[:,:])**p, axis=1))**(1/p)
			a = dist2_nn/dist1_nn
			E[idx_m1] = np.mean(a, axis=0)
			Estar[idx_m1] = np.mean(np.abs(
				x[m1*tau_opt:]-x[ind1_nn+m1*tau_opt]), axis=0)
		E1 = E[1:]/E[:-1]
		E2 = Estar[1:]/Estar[:-1]
		if np.sum(E2<E2_thresh) > 0:
			indE1 = np.argmax(E1>=E1_thresh)
			m_opt = int(m_range[indE1])
		else:
			m_opt = np.nan
	return m_opt, tau_opt


def _calc_autocorr_time(X, n_reps=100):
	n_samples, n_features = X.shape
	n_batches  = int(np.ceil(n_samples**(1/3)))
	sizebatch = int(np.ceil(n_samples**(2/3)))
	autocorr_time_rep_loop = np.zeros((n_reps,n_features))
	autocorr_time_rep = np.zeros((n_reps,n_features))
	for rr in range(n_reps):
		ind_tbatch_start = \
			np.array(np.ceil((n_samples-sizebatch)*\
				 np.random.rand(1,n_batches))-1, dtype='int')[0]
		var_X = np.var(X, axis=0)*n_samples/(n_samples-1)
		Xbatches = np.zeros((n_batches,sizebatch,n_features))
		for bb in np.arange(n_batches):
			Xbatches[bb,:,:] = \
				X[ind_tbatch_start[bb]:ind_tbatch_start[bb]+sizebatch,:]
		mu_Xbatches = np.mean(Xbatches,axis=1)
		var_muXbatches = np.var(mu_Xbatches)*n_batches/(n_batches-1)
		autocorr_time_rep[rr,:] = sizebatch*var_muXbatches/var_X

	autocorr_time = np.mean(autocorr_time_rep, axis=0)
	var_autocorr_time = np.var(autocorr_time_rep, axis=0)*n_reps/(n_reps-1)
	return autocorr_time, var_autocorr_time


def save_pickle(file_path, list_vars, message='Pickling variable: '):
	"""
	file_path: string indicating the absolute path where to save the variables
	list_vars: list containing the variables to be pickled
	"""
	tic = time.time()
	print(message, end='')
	with open(file_path, 'wb') as f:
		pickle.dump(list_vars, f)
	print("%.2f s" %(time.time()-tic))


def get_exceed_idx(data, new, ql=0.98):
	"""
	"""
	pass
