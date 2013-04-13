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
import scipy as sp
import math

def aoa2prop(th,ph):
    """Propogation vectors from angles of arrival. Result is Nx3."""
    return np.array((-np.sin(th)*np.cos(ph), #x
                     -np.sin(th)*np.sin(ph), #y
                     -np.cos(th)             #z
                   )).T # we want Nx3 not 3xN

def aoa2prop_scalar(th,ph):
    """Slightly faster for single arguments, 'cause numpy scalars are slow."""
    return np.array((-math.sin(th)*math.cos(ph), #x
                     -math.sin(th)*math.sin(ph), #y
                     -math.cos(th)               #z
                   ))

def awgn(sig,snrdb,sigpower=0):
    """Additive white gaussian noise.  Assumes signal power is 0 dBW"""
    if sp.iscomplexobj(sig):
        noise = (sp.randn(*sig.shape) + 1j*sp.randn(*sig.shape))/math.sqrt(2)
    else:
        noise = sp.randn(*sig.shape)
    noisev = 10**((sigpower - snrdb)/20)
    return sig + noise*noisev

def makesamples(antennas, theta, phi, num_samples=1, snr=120):
    """
    Generate sample data for an incident signal from inclination=theta,
    azimuth=phi direction.
    """
    assert antennas.shape[1] == 3 # no single antennas, must be in R3

    propvec = aoa2prop(theta,phi)
    phases = sp.dot(antennas,propvec)
    # add some random phase modulation so signals are uncorrelated
    phases = phases + sp.randn(num_samples,1)*0.1
    return awgn(sp.exp(1j*phases),snr)

def eigsort(eigresult):
    """
    Sort the output of scipy.linalg.eig() in terms of 
    eignevalue magnitude
    """
    ix = sp.argsort(abs(eigresult[0]))
    return ( eigresult[0][ix], eigresult[1][:,ix] )

