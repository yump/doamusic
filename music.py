from __future__ import absolute_import, division, print_function
import scipy as sp
import numpy as np
import itertools
from scipy import linalg
from scipy import misc
import util
import _music

class Estimator:
	"""
	A class to carry state for estimating direction of arrival (DOA) 
	with the Multiple SIgnal Classification algorithm.
	"""
	def __init__(self, antennas, covariance, nsignals=None):
		"""
		Get an Estimator.
		"""
		assert antennas.shape[1] == 3
		assert antennas.dtype == 'float64'
		self.antennas = antennas
		self.numel = antennas.shape[0]
		assert covariance.shape == (self.numel,self.numel)
		self.covar = covariance

		#Get the sorted eigenstructure
		self.eigval, self.eigvec = util.eigsort(linalg.eig(covariance))

		# Try to guess the number of incident signals, if unspecified
		if nsignals:
			assert nsignals < self.numel
			self.nsignals = nsignals
			self.noisedim = self.numel - nsignals
		else:
			shaped = abs(self.eigval)
			self.noisedim = sp.diff(shaped).argmax() + 1
			self.nsignals = self.numel - self.noisedim

		#slice the noise space
		self.noisespace = self.eigvec[:,:self.noisedim]
		print("noisespace size:\t{}".format(self.noisespace.shape))
		self.sigspace = self.eigvec[:,self.noisedim:]

	def eigplot():
		"""
		Plot the eigenvalues on a logarithmic chart, for human appraisal
		of the number of incident signals (and any other interesting
		properties of the data.
		"""
		pass

def spectrum(est,theta,phi):
	"""
	Generate a MUSIC pseudospectrum on the cartesian product of theta and
	phi.  The result is a len(theta) x len(phi) real array.

	Parameters
	----------
	est : Estimator
		input data
	
	theta : numpy.ndarray, 1 dimensional
		Values of theta for which to generate the spectrum.
	
	phi : numpy.ndarray, 1 dimensional
		Values of phi for which to generate the spectrum.
	"""
	ants = est.antennas
	metric = sp.dot(est.noisespace,est.noisespace.T.conj())
	result = np.empty((len(theta),len(phi)))
	for i,th in enumerate(theta):
		for j,ph in enumerate(phi):
			result[i,j] = _pmusic(metric,ants,th,ph)
	return result

def doasearch(est,thetaspan,phispan,iterations=4):
	raise NotImplementedError()

def covar(samples):
	"""
	Calculate the covariance matrix as used by Estimator.  
	
	This is not the same as the Octave/Matlab function cov(), but is instead
	equal to Mean [ sample.H * sample ], where sample is a single sample.
	I.E., it is actually the second moment matrix.

	Parameters
	----------
	samples : K x Numel or Numel x 0 complex ndarray
		Complex samples for each of Numel antennas sampled at K times.

	Returns
	-------
		return : Numel x Numel complex ndarray
			Second moment matrix for complex random vector samples.  Used by
			Estimator.
	"""
	samples = sp.asmatrix(samples)
	return ( (samples.H * samples) / samples.shape[0] )

def _pmusic(metric,antennas,theta,phi):
	steer = sp.dot(antennas,util.aoa2prop_scalar(theta,phi))
	#print("ants:\t{}\nsteer:\t{}\nmetric:\t{}".format(antennas.shape,steer.shape,metric.shape))
	return 1/sp.dot(sp.dot(steer.conj(),metric,),steer)
