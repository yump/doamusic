from __future__ import absolute_import, division, print_function
import scipy as sp
import numpy as np
from scipy import linalg
from scipy import misc
import _music

class Estimator:
	"""
	A class to carry state for estimating direction of arrival (DOA) 
	with the Multiple SIgnal Classification algorithm.
	"""
	def __init__(self, antennas, covariance, nsignals=None):
		assert antennas.shape[1] == 3
		assert antennas.dtype == 'float64'
		self.antennas = antennas
		self.numel = antennas.shape[0]
		assert covariance.shape == (numel,numel)
		self.covar = covariance

		#Get the sorted eigenstructure
		self.eigval, self.eigvec = eigsort(linalg.eig(covariance))

		# Try to guess the number of incident signals, if unspecified
		if nsignals:
			assert nsignals < numel
			self.nsignals = nsignals
			self.noisedim = numel - nsignals
		else:
			shaped = abs(eigval)
			self.noisedim = sp.diff(shaped).argmax() + 1
			self.nsignals = numel - noisedim

		#slice the noise space
		self.noisespace = eigvec[:noisedim]
		self.sigspace = eigvec[noisedim:]

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
	phi, which are both 1-d arrays.  The result is a len(theta) x len(phi) 
	real array.
	"""
	



def covar(samples):
	"""
	Calculate the covariance matrix as used by Estimator.  This is not the
	same as the Octave/Matlab function cov(), but is instead equal to 
	Mean [ sample.H * sample ], where sample is a single sample. Samples is
	a K x Numel array, where K is the number of samples and Numel is the 
	number of antennas.
	"""
	samples = sp.asmatrix(samples)
	return ( (samples.H * samples) / samples.shape[0] )


def eigsort(eigresult):
	"""Sort the output of scipy.linalg.eig() in terms of 
	eignevalue magnitude"""

	ix = sp.argsort(abs(eigresult[0]))
	return ( eigresult[0][ix], eigresult[1][:,ix] )

