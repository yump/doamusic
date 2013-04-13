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
        self.sigspace = self.eigvec[:,self.noisedim:]
        print("Noise space dimension: {}".format(self.noisespace.shape))

    def eigplot():
        """
        Plot the eigenvalues on a logarithmic chart, for human appraisal
        of the number of incident signals (and any other interesting
        properties of the data.
        """
        pass

def spectrum(est,
             (theta_lo,theta_hi,theta_sz),
             (phi_lo,phi_hi,phi_sz),
             method=_music.spectrum
             ):
    """
    Generate a MUSIC pseudospectrum on the region specified. The result 
    is a theta_sz x phi_sz real numpy.ndarray. Range specifications are
    inclusive, like linspace.

    Parameters
    ----------
    est : Estimator
        input data
    
    theta_lo,phi_lo : float
        Specify the top-left [0,0] corner of the result.

    theta_hi,phi_hi : float
        Specify the bottom-right [theta_sz-1,phi_sz-1] corner of the result.

    theta_sz, phi_sz : int
        Specify the size of the result

    method : callable
        Choose between the python or cython low-level implementations.  Used
        to check correctness.
    """
    # Wraps either _spectrum or _music.spectrum and provides parallel
    # evaluation.

    # precalculate static arguments as comlpex double and prepare output array
    ants = est.antennas.astype(complex)
    metric = sp.atleast_2d(
                est.noisespace.dot( est.noisespace.T.conj() )
             ).astype(complex)
    result = np.empty((theta_sz,phi_sz))

    # step sizes
    thstep = (theta_hi-theta_lo)/(theta_sz-1)
    phstep = (phi_hi-phi_lo)/(phi_sz-1)

    method(
           metric,
           ants,
           result,
           theta_lo,thstep,theta_sz,
           phi_lo,phstep,phi_sz
    )
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
    steer = sp.exp( 1j*antennas.dot(util.aoa2prop_scalar(theta,phi)) )
    return 1.0 / steer.conj().dot(metric).dot(steer).real

def _spectrum(metric,
              antennas,
              out,
              thlo,thstep,thsz,
              phlo,phstep,phsz
              ):
    # Lower-level spectrum calculator with preprocessed arguments and 
    # pass-by-reference output array, for easier implementation with
    # cython and being farmed out to multiple processes. (The problem is
    # embarassingly parallel.
    assert out.shape == (thsz,phsz)
    for i in xrange(thsz):
        th = thlo + i*thstep
        for j in xrange(phsz):
            ph = phlo + j*phstep
            out[i,j] = _pmusic(metric,antennas,th,ph)
