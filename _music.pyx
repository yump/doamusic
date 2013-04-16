#cython: profile=False
#   Copyright 2013 Russell Haley
#   (Please add yourself if you make changes)
#
#   This file is part of doamusic.
#
#   doamusic is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   doamusic is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with doamusic.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
from numpy import pi
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.math cimport sin,cos
from libc.stdlib cimport abort, malloc, free

np.import_array()

cdef extern from "cmusic.h":
    double cpmusic(
        double complex *metric,
        double complex *antennas,
        double complex *work,   
        size_t N,
        double theta,
        double phi
    ) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def spectrum(
    np.ndarray[complex,ndim=2] metric,
    np.ndarray[complex,ndim=2] ants,
    np.ndarray[double,ndim=2] out,
    double thlo, double thstep, Py_ssize_t thsz,
    double phlo, double phstep, Py_ssize_t phsz
):
    cdef Py_ssize_t i,j
    
    # Ensure inputs contiguous.
    metric = np.ascontiguousarray(metric)
    ants = np.ascontiguousarray(ants)

    cdef double th,ph

    with nogil, parallel():
        # Allocate thread-local workspace buffer. (see cmusic.c for size info)
        work = <complex *> malloc(sizeof(complex)*(2*ants.shape[0]+3))
        if work == NULL:
            abort()

        for i in prange(thsz,schedule='static'):
            th = thlo + i*thstep
            for j in range(phsz):
                ph = phlo + j*phstep
                out[i,j] = cpmusic(
                    &metric[0,0],
                    &ants[0,0],
                    &work[0],
                    ants.shape[0],
                    th,ph
                )
        free(work)

def hillclimb(
    np.ndarray[complex,ndim=2] metric,
    np.ndarray[complex,ndim=2] ants,
    double theta, double phi,
    double tol,
    double step=pi/8.0,
    double scale=0.5
):
    """
    Maximizer using the hill-climbing algorithm to find a local maximum of
    pmusic.

    Parameters
    ----------
    metric : NxN complex numpy.ndarray
        Common subexpression in pmusic calculation.  Equal to the product of
        the noisespace matrix with it's hermitian transpose.

    ants : Nx3 complex numpy.ndarray
        Physical positions of the antennas.  The Nth row is the [x,y,z]
        location of the Nth antenna.

    theta,phi : double
        Where to start the search.

    tol : double
        Return when the step size drops below this. 

    step : double
        Initial step size.

    scale : double
        When none of the neighbors are better with the current step size,
        multiply the step size by this and try some more.
    """
    # Allocate workspace buffer. (see cmusic.c for size info)
    work = <complex *> malloc(sizeof(complex)*(2*ants.shape[0]+3))
    if work == NULL:
        abort()
    
    # Set up intial state
    old = (theta,phi)
    new = old
    old_merit = cpmusic(
        &metric[0,0],
        &ants[0,0],
        &work[0],
        ants.shape[0],
        old[0],old[1]
    )
    new_merit = old_merit
    
    while step > tol:
        for dim in range(len(old)):
            for delta in (-step, +step):
                # perturb one of the dimensions of the point
                cur = old[:dim] + (old[dim]+delta,) + old[dim+1:]
                cur_merit = cpmusic(
                    &metric[0,0],
                    &ants[0,0],
                    &work[0],
                    ants.shape[0],
                    cur[0],cur[1]
                )
                if cur_merit > new_merit and cur_merit > old_merit:
                    new = cur
                    new_merit = cur_merit
        # reduce the step size if we didn't find a better point
        if new == old:
            step = step * scale
        else:
            old = new
            old_merit = new_merit

    free(work)
    return old_merit,old
