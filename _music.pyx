#cython: profile=False
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin,cos


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double pmusic(np.ndarray[complex,ndim=2] metric,
	                np.ndarray[double,ndim=2] antennas,
					np.ndarray[double,ndim=1] propvec, #reusable buffer
	                double theta,
	                double phi):
	# get propogation vector a(th,ph)
	propvec[0] = -sin(theta) * cos(phi)
	propvec[1] = -sin(theta) * sin(phi)
	propvec[2] = -cos(theta)
	# steering matrix 
	steer = np.exp(1j*np.dot(antennas,propvec))
	# Pmusic
	return 1.0 / steer.conj().dot(metric).dot(steer).real

@cython.boundscheck(False)
@cython.wraparound(False)
def spectrum(np.ndarray[complex,ndim=2] metric,
	         np.ndarray[double,ndim=2] antennas,
			 np.ndarray[double,ndim=2] out,
	         double thlo, double thstep, Py_ssize_t thsz,
	         double phlo, double phstep, Py_ssize_t phsz
	         ):
	# reusable buffer for propogation vectors
	cdef np.ndarray[double,ndim=1] propvec = np.empty(3)
	cdef Py_ssize_t i,j
	for i in range(thsz):
		th = thlo + i*thstep
		for j in range(phsz):
			ph = phlo + j*phstep
			out[i,j] = pmusic(metric,antennas,propvec,th,ph)

