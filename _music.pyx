cimport numpy as np
cimport cython
from libc.math cimport sin,cos

def spectrum(est,
		np.ndarray[double,ndim=1] vtheta,
		np.ndarray[double,ndim=1] vphi):

	# preallocate
	cdef np.ndarray[double,ndim=2] result
	result = np.empty(( vtheta.shape[0],vphi.shape[0] ))
	cdef np.ndarray[double,ndim=1] propvec = np.empty(3)
	steermtx = np.asmatrix(np.empty(est.antennas.shape[0]))

	# slurp needed data
	cdef np.ndarray[double,ndim=2] antennas = est.antennas 
	noisespace = est.noisespace

	# precalculate associable multiplication
	metric = noisespace * noisespace.H

	cdef int thx,phx

	for thx in xrange(0,vtheta.shape[0]-1):
		for phx in xrange(0,vphi.shape[0]-1):
			# get propogation vector a(th,ph)
			propvec[0] = -sin(vtheta[thx]) * cos(vphi[phx])
			propvec[1] = -sin(vtheta[thx]) * sin(vphi[phx])
			propvec[2] = -cos(vtheta[thx])
			# steering matrix 
			steermtx = np.asmatrix(np.dot(antennas,propvec))
			# Pmusic
			result[thx,phx] = 1.0 / (steermtx.H * metric * steermtx)

	return result
