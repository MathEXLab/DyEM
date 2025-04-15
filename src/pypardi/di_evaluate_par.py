"""
This script approximates the local dimension and predictibility based on the statistics od the training set.
"""
import numpy as np
import time
from tqdm import tqdm
import scipy.stats as sc
import statsmodels.api as sm
from itertools import combinations_with_replacement




def _idx_self_vars(n_vars):
	idx_self_vars = [0]
	vv = 0
	for ii in range(n_vars-1):
		vv += n_vars-ii
		idx_self_vars.append(vv)
	return idx_self_vars

def _calc_epoch(
	X, tt, vars=[(0,0)], axis=1, p=2, ql=0.98, n_exceeds=2,
	theta_fit='sueveges', p_value=None, exp_test='anderson',
	dql=None, p_cross=None, dim_out=1, comm=None):
	"""

	"""
	if comm is not None:
		rank = comm.Get_rank()
		size = comm.Get_size()
	else:
		rank = 0

	## get dimensions
	n_samples, _, n_vars = X.shape
	n_samples = n_samples - 1	# exclude the new data snapshot

	## calculate delta (both for serial and data_distributed)
	delta = _calc_delta(X=X, tt=tt, p=p, axis=axis, p_cross=p_cross,
						n_samples=n_samples,vars=vars,n_vars=n_vars,
						dim_out=dim_out, comm=comm)

	## switch parallel vs serial computation
	if comm is not None:
		comm.Barrier()

	if rank == 0:
		d = np.zeros([1,dim_out])
		theta = np.zeros([1,dim_out])
		exceeds = np.zeros([np.max(n_exceeds),dim_out])*np.nan
		idx_exceeds = np.zeros([np.max(n_exceeds),dim_out])*np.nan
		H0 = np.zeros([1,dim_out])
		ql_opt = np.zeros([1,dim_out])
		for dd, vv in enumerate(vars):
			res = _calc_d_theta_exceeds_idx(
				delta=delta[:,dd],
				ql=ql[dd],
				n_samples=n_samples,
				n_exceeds=n_exceeds[dd],
				theta_fit=theta_fit,
				p_value=p_value,
				exp_test=exp_test,
				dql=dql,
				comm=comm)
			d[0,dd] = res['d']
			theta[0,dd] = res['theta']
			n_exceeds_new = len(res['exceeds'])
			exceeds[:n_exceeds_new,dd] = res['exceeds']
			idx_exceeds[:n_exceeds_new,dd] = res['idx_exceeds']
			if p_value is not None:
				H0[0,dd] = res['H0']
				if dql is not None:
					ql_opt[0,dd] = res['ql_opt']
	else:
		d = np.zeros([1,dim_out])*np.nan
		theta = np.zeros([1,dim_out])*np.nan
		exceeds = np.zeros([np.max(n_exceeds),dim_out])*np.nan
		idx_exceeds = np.zeros([np.max(n_exceeds),dim_out])*np.nan
		if p_value is not None:
			H0 = np.zeros([1,dim_out])*np.nan
			if dql is not None:
				ql_opt = np.zeros([1,dim_out])*np.nan

	## if parallel wait and broadcast
	if comm is not None:
		comm.Barrier()

		# To be sure that the output is the right one, no matter
		# who is the last rank, broadcast the d, theta and exceeds
		# from leading rank 0 to all the others
		d = comm.bcast(d, root=0)
		theta = comm.bcast(theta, root=0)
		exceeds = comm.bcast(exceeds, root=0)
		idx_exceeds = comm.bcast(idx_exceeds, root=0)

	## collect results
	results = {
		'd': d,
		'theta': theta,
		'exceeds': exceeds,
		'idx_exceeds': idx_exceeds,
	}

	## add p_value to results if required
	if p_value is not None:
		results['H0'] = H0
		if dql is not None:
			results['ql_opt'] = ql_opt
	return results

def _calc_delta(X, tt, n_samples, p=2, axis=1, p_cross=None, vars=[(0,0)],\
				n_vars=1, dim_out=1, comm=None):
	if comm is None:
		dim_out = len(vars)
		delta_self      = np.zeros([n_samples,n_vars])
		norm_delta_self = np.zeros([n_vars,1])
		for vv in range(n_vars):
			delta_self[:,vv] = np.sum(np.abs(X[tt,:,vv] - X[:-1,:,vv])**p,	# exclude the last snapshot, which is the appended new data
											axis=axis)**(1/p)
			norm_delta_self[vv] = np.sum(np.abs(delta_self[:,vv])**p)**(1/p)
			delta_self[:,vv] = delta_self[:,vv] / norm_delta_self[vv]

		delta = np.zeros([n_samples,dim_out])
		for dd,vv in enumerate(vars):
			var1 = vv[0]
			var2 = vv[1]
			if var1 == var2:
				delta[:,dd] = delta_self[:,var1]
			else:
				delta[:,dd] = (np.abs(delta_self[:,var1])**p_cross +
							   np.abs(delta_self[:,var2])**p_cross)**(1/p_cross)
		return delta
	else:
		from mpi4py import MPI
		rank = comm.Get_rank()
		size = comm.Get_size()

		## calculate distance between x and Y
		st = time.time()

		n_samples, _, n_vars = X.shape

		# On each processor, we can calculate the
		# sum of the terms belonging to the processor
		delta_self      = np.zeros([n_samples,n_vars])
		norm_delta_self = np.zeros([n_vars,1])
		for vv in range(n_vars):
			sum_abs_dist_p = np.sum(np.abs(X[tt,:,vv] - X[:-1,:,vv])**p, axis=1)	# exclude the last snapshot, which is the appended new data

			# To sum the results of all processors,
			# we use rank 0 as leading rank
			if rank == 0:
				# only processor 0 will actually get the data
				sum_abs_dist_p_reduced = \
							np.zeros_like(sum_abs_dist_p)
			else:
			    sum_abs_dist_p_reduced = None

			# This is the actual sum of the sum_abs_dist_p
			# calculated at all theprocessors
			comm.Reduce(sum_abs_dist_p, sum_abs_dist_p_reduced, op = MPI.SUM, root = 0)

			# Wait for all processors
			comm.Barrier()

			if rank == 0:
				delta_self[:,vv] = sum_abs_dist_p_reduced**(1/p)
				norm_delta_self[vv] = np.sum(np.abs(delta_self[:,vv])**p)**(1/p)
				delta_self[:,vv] = delta_self[:,vv] / norm_delta_self[vv]

		comm.Barrier()
		if rank == 0:
			delta = np.zeros([n_samples,dim_out])
			for dd,vv in enumerate(vars):
				var1 = vv[0]
				var2 = vv[1]
				if var1 == var2:
					delta[:,dd] = delta_self[:,var1]
				else:
					delta[:,dd] = (np.abs(delta_self[:,var1])**p_cross +
								   np.abs(delta_self[:,var2])**p_cross) \
								   **(1/p_cross)
			return delta

def _calc_d_theta_exceeds_idx(
	delta, n_samples, ql=0.98, n_exceeds=2, theta_fit='sueveges', \
	p_value=None, exp_test='anderson', dql=None, comm=None):

	if comm is None:
		rank = 0
	else:
		rank = comm.Get_rank()

	if rank == 0:

		## calculate negative log dist
		g = -np.log(delta)

		flag_q = False
		ql_new = ql
		n_exceeds_new = n_exceeds
		while flag_q == False:

			## extract only negative log dist that have finite values
			g_finite = g[np.isfinite(g)]

			## sort neg log dist with finite elements
			g_finite_sorted = np.sort(g_finite)

			## calc quantile as with mquantiles with alphap=0.5 and betap=0.5
			aleph = np.array(n_samples * ql_new + 0.5)
			k = np.floor(aleph.clip(1,n_samples-1)).astype(int)
			gamma = (aleph - k).clip(0,1)
			if g_finite_sorted.shape[0] == k:
				ql_new += dql
				if ql_new >= 1:
					flag_q = True
				continue
			else:
				q = (1. - gamma) * g_finite_sorted[k-1] + \
								gamma * g_finite_sorted[k]

			## get the n_exceeds nearest neighbors (i.e., those
			## in the higer quantiles). In this way we do not
			## face the problem encountered with mquantiles
			## when q is equal to one or more of the g values
			g_over_q = g_finite_sorted[-n_exceeds_new:]

			## g indexes of the exceedances (needed to calculate theta)
			## Using g and not g_finite because also the point itself
			## at time t is used
			ids = np.arange(g.shape[0]).reshape([-1,1])
			idx = ids[g > q][-n_exceeds_new:]

			## compute exceedances
			exceeds = g_over_q - q
			idx_exceeds = idx[:,0]

			## fit exponential distribution
			if p_value is not None:
				H0 = _expon_test(
					exceeds=exceeds, p_value=p_value, exp_test=exp_test)
				if H0:
					flag_q = True
				else:
					## update ql if requested and if necessary
					if dql is not None:
						ql_new += dql
						n_exceeds_new = int((1 - ql_new) * n_samples) - 1
						if ql_new >= 1:
							flag_q = True
					else:
						flag_q = True
			else:
				flag_q = True
		ql_opt = ql_new

		## compute dimension
		d = 1 / np.mean(exceeds)

		## compute theta
		if theta_fit == 'ferro':
			theta = _theta_ferro(idx)
		else:
			theta = _theta_sueveges(idx, ql)

		if ql_new > 1:
			d = np.nan
			theta = np.nan
			exceeds = np.array([np.nan,])
			idx_exceeds = np.array([np.nan,])
	else:
		d = np.nan
		theta = np.nan
		exceeds = np.nan
		idx_exceeds = np.nan

	results = {
		'd': d,
		'theta': theta,
		'exceeds': exceeds,
		'idx_exceeds': idx_exceeds,
	}
	if p_value is not None:
		results['H0'] = H0
		if dql is not None:
			results['ql_opt'] = ql_opt
	return results

def _expon_test(exceeds, p_value, exp_test):
	if exp_test=='anderson':
		from scipy.stats import anderson
		if p_value==0.15:
			ind_p_value_anderson = 0
		elif p_value==0.1:
			ind_p_value_anderson = 1
		elif p_value==0.05:
			ind_p_value_anderson = 2
		elif p_value==0.025:
			ind_p_value_anderson = 3
		elif p_value==0.01:
			ind_p_value_anderson = 4
		else:
			raise ValueError(
				'p_value must be one of the following values: ',
				'0.15'' 0.10, 0.05, 0.025, 0.01')

		## perform anderson test
		anderson_stat, anderson_crit_val, anderson_sig_lev = \
			anderson(exceeds, dist='expon')
		if anderson_stat > anderson_crit_val[ind_p_value_anderson]:
			H0 = False # Reject H0
		else: H0 = True # Do not reject H0

	elif exp_test=='chi2':
		pplot = sm.ProbPlot(exceeds, sc.expon)
		xq = pplot.theoretical_quantiles
		yq = pplot.sample_quantiles
		p_fit = np.polyfit(xq, yq, 1)
		yfit = p_fit[0] * xq + p_fit[1]

		## perform Chi-Square Goodness of Fit Test
		chi2, p_chi2 = sc.chisquare(f_obs=yq, f_exp=yfit)
		if p_chi2 < p_value: H0 = False # Reject H0
		else: H0 = True # Do not reject H0
	return H0

def _theta_ferro(idx):
	Ti = idx[1:] - idx[:-1]
	if np.max(Ti) > 2:
		res = 2 * (np.sum(Ti - 1)**2) / \
			((Ti.size - 1) * np.sum((Ti - 1) * (Ti - 2)))
	else:
		res = 2 * (np.sum(Ti)**2) / ((Ti.size - 1) * np.sum(Ti**2))
	return min(1, res)

def _theta_sueveges(idx , ql):
	q = 1 - ql
	Ti = idx[1:] - idx[:-1]
	Si = Ti - 1
	Nc = np.sum(Si > 0)
	K  = np.sum(q * Si)
	N  = Ti.size
	return (K + N + Nc - np.sqrt((K + N + Nc)**2 - 8 * Nc * K)) / (2 * K)

def _check_inputs(X, new, ql=0.98, p_cross=None):
	"""
	Checks that data format has correct shape,
	and assigns consistent axis for dask operations
	(inconsistent with numpy).
	X: train set data (n_samples, n_features, n_vars)
	new: output data (n_samples, n_features, n_vars)
	"""
	## make sure X is a 3D matrix
	if X.ndim == 2:
		X = X.reshape(X.shape + (1,))
	if X.ndim != 3:
		raise ValueError('Input matrix `X` must be 3D.')

	n_vars = X.shape[2]
	if p_cross is not None:
		V = list(range(n_vars))
		vars = [comb for comb in combinations_with_replacement(V, 2)]
	else:
		vars = []
		for vv in range(n_vars):
			vars.append((vv,vv))
	dim_out = len(vars)

	## make sure new is a 3D matrix
	if new is None:
		pred_length = 0
	else:
		if new.ndim == 2:
			new = new.reshape(new.shape + (1,))
		if new.ndim != 3:
			raise ValueError('Input matrix `new` must be 3D.')
		pred_length = new.shape[0]

	if type(ql) is not list:
		ql = [ql]*dim_out
	else:
		if len(ql) != dim_out:
			raise ValueError('The list `ql` must have same shape of dim_out.')

	# ONLY support NUMPY now
	# ## for dask operation (axis not consistent with numpy)
	# if isinstance(X, xr.DataArray):
	# 	axis = 0
	# else:
	# 	axis = 1
	axis = 1
	return X, new, axis, vars, dim_out, ql, pred_length

def compute(
		X, new, ql=0.98, p=2, theta_fit="sueveges",
		p_value=None, dql=None, exp_test='anderson',
		p_cross=None, distributed='none',comm=None, **kwargs):
	'''
	for i in range(pred_length):
		calculate the DI for the i-th prediction
	Take average of all the DI
	'''
	# not using parallel
	if comm is not None:
		## get rank and sync all MPI operations
		rank = comm.Get_rank()
		comm.Barrier()
	else:
		rank = 0
	
	## float type to be used
	float_type = np.float64

	# prepare data
	X, new, axis, vars, dim_out, ql, pred_length = _check_inputs(X, new, ql=ql, p_cross=p_cross)

	## get dimensions
	n_samples, n_features, n_vars = X.shape	# we don't need, only need n_pred
	n_samples = n_samples + 1	# add the new data snapshot

	## calc number of exceeds
	n_exceeds = np.zeros([dim_out,], dtype=int)
	for qq in range(dim_out):
		n_exceeds[qq] = int((1 - ql[qq]) * n_samples) - 1
		if n_exceeds[qq] < 2:
			print('Not enough epochs to perform the analysis properly.')
			print('Setting n_exceeds=2 for vars idx ', qq)
			n_exceeds[qq] = 2

	## initialize vectors
	max_n_exceeds = np.max(n_exceeds)
	if pred_length != 0:
		d       = np.zeros([pred_length,dim_out], dtype=float_type)
		theta   = np.zeros([pred_length,dim_out], dtype=float_type)
		H0      = np.zeros([pred_length,dim_out], dtype=bool)
		exceeds = \
			np.zeros([pred_length,max_n_exceeds,dim_out], dtype=float_type)*np.nan
		idx_exceeds = \
			np.zeros([pred_length,max_n_exceeds,dim_out], dtype=float_type)*np.nan
		ql_opt = np.zeros([pred_length,dim_out], dtype=float_type)
		if p_cross is not None:
			alpha = np.ones([pred_length,dim_out], dtype=float_type)
	else: 
		pred_length = n_samples
		d       = np.zeros([pred_length,dim_out], dtype=float_type)
		theta   = np.zeros([pred_length,dim_out], dtype=float_type)
		H0      = np.zeros([pred_length,dim_out], dtype=bool)
		exceeds = \
			np.zeros([pred_length,max_n_exceeds,dim_out], dtype=float_type)*np.nan
		idx_exceeds = \
			np.zeros([pred_length,max_n_exceeds,dim_out], dtype=float_type)*np.nan
		ql_opt = np.zeros([pred_length,dim_out], dtype=float_type)
		if p_cross is not None:
			alpha = np.ones([pred_length,dim_out], dtype=float_type)

	## loop over pred_length
	
	for tt in tqdm(range(pred_length), desc='# iterations on epochs (serial)'):
		# print(X.shape)
		# print(new[tt:tt+1,...].shape)
		temp = np.concatenate((X, new[tt:tt+1,...]), axis=0)

		res = _calc_epoch(
			X=temp,
			tt=n_samples-1,	# only loop over the last appended data
			# tt = tt,
			vars=vars,
			axis=axis,
			p=p, ql=ql,
			n_exceeds=n_exceeds,
			theta_fit=theta_fit,
			p_value=p_value,
			exp_test=exp_test,
			dql=dql,
			p_cross=p_cross,
			dim_out=dim_out,
			comm=comm)

		if rank == 0:
			d[tt,:] = res['d']
			theta[tt,:] = res['theta']
			for dd in range(dim_out):
				n_exceeds[dd] = len(res['exceeds'][:,dd])
				exceeds[tt,:n_exceeds[dd],dd] = res['exceeds'][:,dd]
				idx_exceeds[tt,:n_exceeds[dd],dd] = res['idx_exceeds'][:,dd]

			if p_value is not None:
				H0[tt,:] = res['H0']
				if dql is not None:
					ql_opt[tt,:] = res['ql_opt']

			if p_cross is not None:
				idx_self_vars = _idx_self_vars(n_vars)
				vars_self = [comb \
					for comb in combinations_with_replacement(idx_self_vars, 2)]
				dd = -1
				for var_self in vars_self:
					dd += 1
					var1 = var_self[0]
					var2 = var_self[1]
					num = len(np.intersect1d(
								idx_exceeds[tt,:n_exceeds[var1],var1],
								idx_exceeds[tt,:n_exceeds[var2],var2]))
					alpha[tt,dd] = num / n_exceeds[var1]

	## add mandatory results
	results = {
		'd': d,
		'theta': theta,
		'exceeds': exceeds,
		'idx_exceeds': idx_exceeds,
	}

	## add optional results if required
	if p_value is not None:
		results['H0'] = H0
		if dql is not None:
			results['ql'] = ql_opt
	if p_cross is not None:
		results['alpha'] = alpha

	return results


#=================parallel=================
def time_parallel_loop(
	X, tt, tt_local, vars, n_vars, axis, p, ql, n_exceeds,
	theta_fit, p_value, exp_test, dql, p_cross, dim_out,
	d, theta, exceeds, idx_exceeds, H0, ql_opt, alpha):

	## compute single epoch
	res = _calc_epoch(
		X=X, tt=tt, vars=vars, axis=axis, p=p, ql=ql, n_exceeds=n_exceeds,
		theta_fit=theta_fit, p_value=p_value, exp_test=exp_test, dql=dql,
		p_cross=p_cross, dim_out=dim_out)

	d[tt_local,:] = res['d']
	theta[tt_local,:] = res['theta']
	for dd in range(dim_out):
		n_exceeds[dd] = len(res['exceeds'][:,dd])
		exceeds[tt_local,:n_exceeds[dd],dd] = res['exceeds'][:,dd]
		idx_exceeds[tt_local,:n_exceeds[dd],dd] = res['idx_exceeds'][:,dd]

	if p_value is not None:
		H0[tt_local,:] = res['H0']
		if dql is not None:
			ql_opt[tt_local,:] = res['ql_opt']

	if p_cross is not None:
		idx_self_vars = _idx_self_vars(n_vars)
		vars_self = [comb \
			for comb in combinations_with_replacement(idx_self_vars, 2)]
		dd = -1
		for var_self in vars_self:
			dd += 1
			var1 = var_self[0]
			var2 = var_self[1]
			num = len(np.intersect1d(
				idx_exceeds[tt_local,:n_exceeds[var1],var1],
				idx_exceeds[tt_local,:n_exceeds[var2],var2]))
			alpha[tt_local,dd] = num / n_exceeds[var1]

	## add mandatory results
	results = {
		'd': d,
		'theta': theta,
		'exceeds': exceeds,
		'idx_exceeds': idx_exceeds,
	}

	## add optional results if required
	if p_value is not None:
		results['H0'] = H0
		if dql is not None:
			results['ql'] = ql_opt
	if p_cross is not None:
		results['alpha'] = alpha
	return results

def compute_distributed_time(
	X, new, ql=0.98, p=2, theta_fit="sueveges",
	p_value=None, dql=None, exp_test='anderson',
	p_cross=None, **kwargs):
	"""
	ql: float between 0 and 1 [default: 0.98]
		Quantile threshold to determine the exceeds.
	p_value: float between 0 and 1 or None [default: None]
		If not None, test the null hypothesis H0 for which
		the exceeds follow an exponential distribution
		(GPD with fixed null shape). The used statistical
		test is specified by `exp_test`.
	exp_test: 'anderson' or 'chi2' [default: 'anderson']
		Statistical test used to test the null hypothesis
		H0 for which the exceeds follow an exponential
		distribution (GPD with fixed null shape).
		`anderson` Anderson-Darling test of exponential
		distribution. `chi2` chi-square test on the
		residuals of a best linear fit on the
			q-q plot having on the x-axis a theoretical
			exponential distribution and on the y-axis
			the exceeds sample distribution.
	dql: float between 0 and 1 or None [default: None]
		If not None, perform a statistical test to
		test the null hypothesis H0 for which the exceeds
		follow an exponential distribution (GPD with fixed
		null shape). If H0 is rejected, then update
		ql for a given tested epoch increasing it
		by dql until H0 is accepted. If the updated ql >= 1,
		stop updating and set to np.nan the local indices
		for the tested epoch. Stop updating when H0 cannot
		be rejected and store the updated ql value.
	"""
	# mpi capabilities
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	# if rank == 0:
	# 	print(f'total number of ranks: {size}')

	## float type to be used
	float_type = np.float64

	## check data format
	X, new, axis, vars, dim_out, ql, pred_length = _check_inputs(X, new, ql=ql, p_cross=p_cross)

	## get dimensions
	n_samples, n_features, n_vars = X.shape
	n_new_samples = new.shape[0]

	## calc number of exceeds
	n_exceeds = np.zeros([dim_out,], dtype=int)
	for qq in range(dim_out):
		n_exceeds[qq] = int((1 - ql[qq]) * n_samples) - 1
		if n_exceeds[qq] < 2:
			print('Not enough epochs to perform the analysis properly.')
			print('Setting n_exceeds=2 for vars idx ', qq)
			n_exceeds[qq] = 2

	## compute indices
	# split arrays for mpi
 	# TODO: on new
	perrank = n_new_samples // size
	remaind = n_new_samples % size

	comm.Barrier()
	max_n_exceeds = np.max(n_exceeds)

	if rank == size - 1:
		n_sample_rank = perrank + remaind
	else:
		n_sample_rank = perrank

	## initialize vectors
	d       = np.zeros([n_sample_rank,dim_out], dtype=float_type)
	theta   = np.zeros([n_sample_rank,dim_out], dtype=float_type)
	exceeds = np.zeros([n_sample_rank,max_n_exceeds,dim_out],
		dtype=float_type)*np.nan
	idx_exceeds = np.zeros([n_sample_rank,max_n_exceeds,dim_out],
		dtype=float_type)*np.nan
	if p_value is not None:
		H0 = np.zeros([n_sample_rank,dim_out], dtype=bool)
		if dql is not None:
			ql_opt = np.zeros([n_sample_rank,dim_out], dtype=float_type)
		else: ql_opt = None
	else:
		H0 = None
		ql_opt = None
	if p_cross is not None:
		alpha = np.ones([n_sample_rank,dim_out], dtype=float_type)
	else: alpha = None

	## loop over time snapshots (epochs)
	if rank != size - 1:

		for tt in range(rank * perrank, (rank + 1) * perrank):
            # divide new data
			new_sample = new[rank * perrank: (rank + 1) * perrank, :, :]	# new data to compute per rank
			# arrays are defined locally per rank
			tt_local = tt - rank * perrank 

			temp = np.concatenate((X, new_sample[tt_local:tt_local+1,...]), axis=0)	# each time append a snapshot
			# print(f'rank: {rank}, tt: {tt}, tt_local: {tt_local}, temp shape: {temp.shape}')
			res = time_parallel_loop(	
				X=temp, tt=n_samples, tt_local=tt_local, vars=vars, n_vars=n_vars,
				axis=axis, p=p, ql=ql, n_exceeds=n_exceeds, theta_fit=theta_fit,
				p_value=p_value, exp_test=exp_test, dql=dql, p_cross=p_cross,
				dim_out=dim_out, d=d, theta=theta, exceeds=exceeds,
				idx_exceeds=idx_exceeds, H0=H0, ql_opt=ql_opt, alpha=alpha,
			)

	else:
		## loop over remaining times (remainder)
		pbar = tqdm(total = n_sample_rank, desc='# iterations on epochs')
		for tt in range((size-1)*perrank, n_new_samples):
			pbar.update(1)
			# print(f'n_samples: {n_samples}')
			# divide new data
			new_sample = new[rank * perrank:, :, :]	# new data to compute per rank
			# print(f'rank: {rank}, new_sample shape: {new_sample.shape}')
			# arrays are defined locally per rank
			tt_local = tt - (size-1)*perrank 
			temp = np.concatenate((X, new_sample[tt_local:tt_local+1,...]), axis=0)
			# print(f'rank: {rank}, tt: {tt}, tt_local: {tt_local}, temp shape: {temp.shape}')
			res = time_parallel_loop(
				X=temp, tt=n_samples, tt_local=tt_local, vars=vars, n_vars=n_vars,
				axis=axis, p=p, ql=ql, n_exceeds=n_exceeds, theta_fit=theta_fit,
				p_value=p_value, exp_test=exp_test, dql=dql, p_cross=p_cross,
				dim_out=dim_out, d=d, theta=theta, exceeds=exceeds,
				idx_exceeds=idx_exceeds, H0=H0, ql_opt=ql_opt, alpha=alpha,
			)

	## get mandatory results from ranks and gather
	comm.Barrier()
	d = res['d']
	theta = res['theta']
	exceeds = res['exceeds']
	idx_exceeds = res['idx_exceeds']
	d       = comm.gather(d, root=0)
	theta   = comm.gather(theta, root=0)
	exceeds = comm.gather(exceeds, root=0)
	idx_exceeds = comm.gather(idx_exceeds, root=0)

	## get optional results from ranks and gather
	if p_value is not None:
		H0 = res['H0']
		H0  = comm.gather(H0, root=0)
		if dql is not None:
			ql_opt = res['ql']
			ql_opt = comm.gather(ql_opt, root=0)
	if p_cross is not None:
		alpha = res['alpha']
		alpha = comm.gather(alpha, root=0)

	## rank 0 operations
	if rank == 0:
		## concatenated vectors after gathering
		d       = np.concatenate(d, axis=0)
		theta   = np.concatenate(theta, axis=0)
		# exceeds = np.concatenate(exceeds, axis=0)	# we comment this line because exceeds may exceed the size of mpi message passing, need to fix later
		if p_value is not None:
			H0  = np.concatenate(H0, axis=0)
			if dql is not None:
				ql_opt = np.concatenate(ql_opt, axis=0)
		if p_cross is not None:
			alpha  = np.concatenate(alpha, axis=0)

	## add mandatory results
	results = {
		'd': d,
		'theta': theta,
		'exceeds': exceeds,
		'idx_exceeds': idx_exceeds,
	}

	## add optional results if required
	if p_value is not None:
		results['H0'] = H0
		if dql is not None:
			results['ql'] = ql_opt
	if p_cross is not None:
		results['alpha'] = alpha
	return results, comm