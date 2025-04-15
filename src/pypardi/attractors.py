import os
import sys
import time
import numpy as np
np.random.seed(624)
from math import cos
from math import fabs
from math import sin
from math import sqrt
# integration
from scipy.integrate import solve_ivp
# plotting tools
import matplotlib.pyplot as plt



# Attractors available
# ==============================================================================

def becker(t, state, beta1=1., beta2=0.84,
	rho=0.048, kappaprime=0.8525, nu=4.2164*10**-5):

	kappacr1 = beta1 - 1.0
	kappacr2 = \
		(kappacr1 + rho * (2 * beta1 + (beta2 - 1) * (2 + rho)) + \
		(np.sqrt(4 * rho * rho * (kappacr1 + beta2)) + \
		(kappacr1 + rho * rho * (beta2 - 1)) * \
		(kappacr1 + rho * rho * (beta2 - 1)))) / (2+2*rho)
	kappa = kappaprime * kappacr2

	x, y, z  = state
	ydot = kappa * (1 - np.exp(x))
	zdot = -rho * (beta2 * x + z) * np.exp(x)
	xdot = (((beta1 - 1) * x + y - z) * np.exp(x) + ydot - zdot) \
		/ (1 + nu * np.exp(x))
	return np.array([xdot, ydot, zdot])

def chua(t, state, a=15.6, b=28, mu0=-1.143, mu1=-0.714):
	"""
	Chua attractor.
	Chua circuit. This is a simple electronic circuit that
	exhibits classic chaotic behavior. This means roughly
	that it is a "nonperiodic oscillator".

	Parameters
	==========
	a, b, mu0, mu1 - are chua system parameters.
	Default values are:
		- x0 = 0, y0 = 0, z0=0,
		  a = 15.6, b = 28, mu0 = -1.143, mu1 = -0.714
	"""
	x,y,z = state
	ht = mu1 * x + 0.5 * (mu0 - mu1) * (fabs(x + 1) - fabs(x - 1))
	xdot = a * (y - x - ht)
	ydot = x - y + z
	zdot = -b * y
	return np.array([xdot, ydot, zdot])

def duffing(t,state, a=0.1, b=0.1, omega=1.2):
	"""
	Duffing attractor.

	Parameters
	==========
	a, b - are duffing system parameters.
	Default values are:
		- x0 = 0, y0 = 0, z0 = 0, a = 0.1 and b = 0.1;
		  to verify value b
	"""
	x,y,z = state
	xdot = y
	ydot = -a * y - x ** 3 + b * cos(omega*z)
	zdot = 1
	return np.array([xdot, ydot, zdot])

def duffing_map(t,state, a=2.75, b=0.2):
	"""
	Duffing_map attractor.
	It is a discrete-time dynamical system (2nd order).

	Parameters
	==========
	a, b - are duffing_map system parameters.
	Default values are:
		- x0 = 0, y0 = 0, z0 = 0, a = 2.75 and b = 0.2
	"""
	x,y,z = state
	xdot = y
	ydot = a * y - y ** 3 - b * x
	zdot = 1
	return np.array([xdot, ydot, zdot])

def lorenz(t,state, sigma=10., beta=8/3, rho=28.):
	"""
	Lorenz attractor.
	Lorenz attractor is a 3rd order system.
	In 1963, E. Lorenz developed a simplified
	mathematical model for atmospheric convection.

	Parameters
	==========
	sigma, beta, rho - are lorenz system parameters.
	Default values are:
		- x0 = 0, y0 = 1, z0 = 1.05,
		  sigma = 10, beta = 8/3, rho = 28
	"""
	x,y,z = state
	xdot = sigma * (y - x)
	ydot = x * (rho - z) - y
	zdot = x * y - beta * z
	return np.array([xdot, ydot, zdot])

def lotka_volterra(t, state):
	"""
	Lotka_volterra attractor.
	Lotka_volterra system does not have any system parameters.
	The Lotka–Volterra equations, also known as the predator–prey
	equations, are a pair of first-order nonlinear differential
	equations, frequently used to describe the dynamics of biological
	systems in which two species interact, one as a predator and
	the other as prey.

	Chaotic Lotka-Volterra model require a careful tuning of parameters
	and are even less likely to exhibit chaos as the number of species
	increases. Possible initial values:
		- x0 = 0.6, y0 = 0.2, z0 = 0.01
	"""
	x,y,z = state
	xdot = x * (1 - x - 9 * y)
	ydot = -y * (1 - 6 * x - y + 9 * z)
	zdot = z * (1 - 3 * x - z)
	return np.array([xdot, ydot, zdot])

def nose_hover(t, state):
	"""
	Nose_hover attractor.
	Nose–Hoover system does not have any system parameters.
	The Nose–Hoover thermostat is a deterministic algorithm for
	constant-temperature molecular dynamics simulations. It was
	originally developed by Nose and was improved further by Hoover.

	Nose–Hoover oscillator is a 3rd order system.
	Nose–Hoover system has only five terms and
	two quadratic nonlinearities. Possible initial values:
		- x0 = 0, y0 = 0, z0 = 0
	"""
	x,y,z = state
	r = np.random.randn(3)
	xdot = y
	ydot = y * z - x
	zdot = 1 - y * y
	return np.array([xdot, ydot, zdot])

def rikitake(t, state, a=2, b=3,c=5, d=0.75):
	"""
	Rikitake attractor.
	Rikitake system is a 3rd order system,
	that attempts to explain the reversal
	of the Earth’s magnetic field.

	Parameters
	==========
	a, and mu - are rikitake system parameters.
	Default values are:
		- x0 = 0, y0 = 0, z0 = 0, a = 5, mu = 2

	Another useful combinations is:
		- x0 = 0, y0 = 0, z0 = 0, a = 1, mu = 1
	"""
	x,y,z = state
	xdot = -a * x +  y * (z + c)
	ydot = -b * y + x * (z - c)
	zdot = d * z - x * y
	return np.array([xdot, ydot, zdot])

def rossler(t, state, a=0.2, b=0.2, c=5.7):
	"""
	Rossler attractor.

	Parameters
	==========
	a, b and c - are rossler system parameters.
	Default values are:
		- x0 = 0, y0 = 0, z0 = 0,
		  a = 0.2, b = 0.2 and c = 5.7.

	Other useful combinations are:
	1) x0 = 0, y0 = 0, z0 = 0,
	   a = 0.1, b = 0.1 and c = 14 (another useful parameters)
	2) x0 = 0, y0 = 0, z0 = 0,
	   a = 0.5, b = 1.0 and c = 3 (J. C. Sprott)

	Notes
	=====
	- Varying a:
	b = 0.2 and c = 5.7 are fixed. Change a:

	a <= 0 : Converges to the centrally located fixed point
	a = 0.1: Unit cycle of period 1
	a = 0.2: Standard parameter value selected by Rössler, chaotic
	a = 0.3: Chaotic attractor, significantly more Möbius strip-like
			 (folding over itself).
	a = 0.35: Similar to .3, but increasingly chaotic
	a = 0.38: Similar to .35, but increasingly chaotic

	- Varying b:
	a = 0.2 and c = 5.7 are fixed. Change b:

	If b approaches 0 the attractor approaches infinity,
	but if b would be more than a and c, system becomes
	not a chaotic.

	- Varying c:
	a = b = 0.1 are fixed. Change c:

	c = 4       : period-1 orbit,
	c = 6       : period-2 orbit,
	c = 8.5     : period-4 orbit,
	c = 8.7     : period-8 orbit,
	c = 9       : sparse chaotic attractor,
	c = 12      : period-3 orbit,
	c = 12.6    : period-6 orbit,
	c = 13      : sparse chaotic attractor,
	c = 18      : filled-in chaotic attractor.
	"""
	x,y,z = state
	xdot = -(y + z)
	ydot = x + a * y
	zdot = b + z * (x - c)
	return np.array([xdot, ydot, zdot])

def wang(t, state):
	"""
	Wang attractor.
	Wang system (improved Lorenz model) as classic chaotic attractor.
	Possible initial condition:
		- x0 = 0, y0 = 0, z0 = 0,
	"""
	x, y, z  = state
	xdot = x - y * z
	ydot = x - y + x * z
	zdot = -3 * z + x * y
	return np.array([xdot, ydot, zdot])

# ==============================================================================



# Compute attractor
# ==============================================================================

def _get_rhs(attractor):
	return dict_attractors[attractor]

def compute_attractor(
	attractor, y_init, dt, num_steps, args=None, method='BDF'):
	# Step through "time", calculating the partial derivatives
	# at the current point and using them to estimate the next
	# point. Need one more for the initial values

	t0 = 0.0
	tn = num_steps * dt - t0
	t_eval = np.linspace(t0, tn, num_steps+1)
	rhs = _get_rhs(attractor)
	sol = solve_ivp(
		fun=rhs,
		t_span=[t0, tn],
		t_eval=t_eval,
		y0=y_init,
		args=args,
		method=method,
		dense_output=True
	)
	results = {'t': sol['t'], 'sol': sol['y'].T, 'attractor': attractor}
	return results

# ==============================================================================




# Calculators of attractor properties
# ==============================================================================

def check_min_max(data) -> dict:
	"""
	Calculate minimum and maximum for data coordinates.

	Parameters
	----------
	- data [numpy.ndarray]: data matrix of shape
		n_samples, n_observables
	"""
	min_coord = np.min(data, axis=0)
	max_coord = np.max(data, axis=0)
	dict_minmax = {'min': min_coord, 'max': max_coord}
	return dict_minmax

def check_moments(data, axis=0) -> dict:
	"""
	Calculate stochastic parameters:
	mean, variance, skewness, kurtosis,
	and median.

	Parameters
	----------
	- data [numpy.ndarray]: data matrix of shape
		n_samples, n_observables
	"""
	from scipy.stats import kurtosis
	from scipy.stats import skew
	dict_moments = {
		"mean"    : np.mean  (data, axis=axis),
		"variance": np.var   (data, axis=axis),
		"median"  : np.median(data, axis=axis),
		"skewness": skew     (data, axis=axis),
		"kurtosis": kurtosis (data, axis=axis),
	}
	return dict_moments

def check_probability(data, kde_points=1000):
	"""
	Check probability for each chaotic coordinates.

	Parameters
	----------
	- data [numpy.ndarray]: data matrix of shape
		n_samples, n_observables
	"""
	from scipy.stats import gaussian_kde
	nd = data.shape[1]
	p_axi = np.zeros([kde_points, nd])
	d_kde = np.zeros([kde_points, nd])
	for ii in range(nd):
		p_axi[:,ii] = np.linspace(data[:,ii].min(), data[:,ii].max(), kde_points)
		d_kde[:,ii] = gaussian_kde(data[:,ii]).evaluate(p_axi[:,ii])
		d_kde[:,ii] /= d_kde[ii].max()
	return d_kde

def calculate_spectrum(data, fft_points=4096):
	"""
	Calculate FFT (in dB) for input 3D coordinates.
	You can set number of FFT points into the object instance.

	Parameters
	----------
	- data [numpy.ndarray]: data matrix of shape
		n_samples, n_observables
	"""
	from scipy.fft import fft, fftshift, fftfreq
	spectrum = fft(data, fft_points, axis=0)
	spectrum = np.abs(fftshift(spectrum, axes=0))
	# spectrum = np.abs(spectrum)
	spectrum /= np.max(spectrum)
	spec_log = 20 * np.log10(spectrum + np.finfo(np.float32).eps)
	return spec_log

def calculate_correlation(data):
	"""
	Calculate auto correlation function for chaotic coordinates.

	Parameters
	----------
	- data [numpy.ndarray]: data matrix of shape
		n_samples, n_observables
	"""
	nn, mm = 3, len(data)
	auto_corr = np.zeros([mm,nn])
	for ii in range(nn):
		auto_corr[:,ii] = np.correlate(data[:,ii], data[:,ii], "same")
	return auto_corr

# ==============================================================================



# Plotting tools
# ==============================================================================

def plot_attractor(results):
	"""
	Plot attractor in both phase space, and time.

	Parameters
	----------
	- results [tuple]: containing t, sol, and attractor,
	 	where
		`t`: np.ndarray of shape n_samples (time),
		`sol`: np.ndarray of shape n_samples, n_observables (solution),
		`attractor`: str with attractor name
	"""
	t = results['t']
	sol = results['sol']
	attractor = results['attractor']
	fig = plt.figure()
	ax  = plt.axes(projection='3d')
	ax.plot3D(sol[:,0], sol[:,1], sol[:,2], 'green')
	ax.set_title(attractor)
	plt.show()

	fig, axs = plt.subplots(3,1, sharex=True)
	fig.suptitle(attractor)
	axs[0].plot(t, sol[:,0])
	axs[1].plot(t, sol[:,1])
	axs[2].plot(t, sol[:,2])
	plt.show()

# ==============================================================================

dict_attractors = {
	'becker'        : becker,
	'chua'          : chua,
	'duffing'       : duffing,
	'duffing_map'   : duffing_map,
	'lorenz'        : lorenz,
	'rossler'       : rossler,
	'lotka_volterra': lotka_volterra,
	'nose_hover'    : nose_hover,
	'rikitake'      : rikitake,
	'wang'          : wang,
}

dict_parameters = {
	"chua": {
		"a"  : [15.6,15.6],
		"b"  : [25,51],
		"mu0": [-1.143,-1.143],
		"mu1": [-0.714,-0.714]
	},
	"duffing": {
		"a": [0.1,0.1],
		"b": [0.1,0.65]
	},
	"lorenz": {
		"sigma": [10,10],
		"beta" : [8/3,8/3],
		"rho"  : [28,100]
	},
	"rikitake": {
		"a": [2,7],
		"b": [3,3],
		"c": [5,5],
		"d": [0.75,0.75]
	},
	"rossler": {
		"a": [0.2,0.2],
		"b": [0.2,0.2],
		"c": [4,18]
	},
	"becker":{
		"beta1"     : [1.0,1.0],
		"beta2"     : [0.84,0.84],
		"rho"       : [0.048,0.3],
		"kappaprime": [0.8525,0.922],
		"nu"        : [4.2164*10**-5,4.2164*10**-5]
	}
}
