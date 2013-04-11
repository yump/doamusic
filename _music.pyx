import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin,cos

@cython.boundscheck(False)
@cython.wraparound(False)

def pmusic(np.ndarray[complex,ndim=2] metric not None,
	       np.ndarray[double,ndim=2] antennas not None,
	       double theta,
	       double phi):

	# preallocate
	cdef np.ndarray[double,ndim=1] propvec = np.empty(3)

	# get propogation vector a(th,ph)
	propvec[0] = -sin(theta) * cos(phi)
	propvec[1] = -sin(theta) * sin(phi)
	propvec[2] = -cos(theta)
	# steering matrix 
	steer = np.exp(1j*np.dot(antennas,propvec))
	# Pmusic
	return 1.0 / np.dot( np.dot(steer.conj(),metric ), steer).real

