import scipy as sp
from scipy import misc

class estimator
	"""
	A class to estimate direction of arrival (DOA) using the Multiple
	SIgnal Classification algorithm.
	"""
	def __init__(self, antennas):
		assert antennas.shape[1] == 3
		assert antennas.dtype == 'float64'
		self.antennas = antennas
		self.numel = antennas.shape[0]
		self.numsamples = 0
		self.covar = sp.matrix(sp.zeros((numel,numel)))

	def newsamples(self,samples):
		"""
		Insert a N x num_antennas matrix of samples.
		Each row is an iteration.
		"""
		assert samples.shape[1] == 3
		
			
