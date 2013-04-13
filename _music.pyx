#cython: profile=False
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin,cos

np.import_array()

cdef extern from "cmusic.h":
    double cpmusic(
                   double complex *metric,
                   double complex *antennas,
                   double complex *work,   
                   size_t N,
                   double theta,
                   double phi
                  )

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double pmusic(np.ndarray[complex,ndim=2] metric,
                    np.ndarray[complex,ndim=2] antennas,
                    np.ndarray[complex,ndim=1] propvec, #reusable buffer
                    np.ndarray[complex,ndim=1] steer,   #reusable buffer
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
             np.ndarray[complex,ndim=2] ants,
             np.ndarray[double,ndim=2] out,
             double thlo, double thstep, Py_ssize_t thsz,
             double phlo, double phstep, Py_ssize_t phsz
             ):
    # Allocate workspace buffer.
    cdef np.ndarray[complex,ndim=1] work = np.empty(ants.shape[0]*2 +3,complex)
    
    # Ensure inputs contiguous.
    metric = np.ascontiguousarray(metric)
    ants = np.ascontiguousarray(ants)

    cdef Py_ssize_t i,j
    for i in range(thsz):
        th = thlo + i*thstep
        for j in range(phsz):
            ph = phlo + j*phstep
            out[i,j] = cpmusic(&metric[0,0],
                               &ants[0,0],
                               &work[0],
                               ants.shape[0],
                               th,ph)

    

