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

    antennas = antennas * (2*sp.pi) # convert from wavelengths to radians
    propvec = aoa2prop(theta,phi)
    phases = sp.dot(antennas,propvec)
    # add some random phase modulation so signals are uncorrelated
    phases = phases + sp.randn(num_samples,1)*0.1
    return sp.exp(1j*phases)

def eigsort(eigresult):
    """
    Sort the output of scipy.linalg.eig() in terms of 
    eignevalue magnitude
    """
    ix = sp.argsort(abs(eigresult[0]))
    return ( eigresult[0][ix], eigresult[1][:,ix] )

def sph2cart(sph):
    """
    Convert one or more spherical coordinates to cartesian.

    Parameters
    ----------
    sph : shape 3 or Nx3 numpy.ndarray 
        Spherical coordinates of the form (r,theta,phi), where theta is the
        inclination angle from the Z axis and phi is the azimuth angle from the
        X axis.

    Returns
    -------
    cart : shape 3 or Nx3 numpy.ndarray
        Cartesian coordinates of the form (x,y,z)
    """
    sph = np.atleast_2d(sph).T
    x = sph[0] * np.sin(sph[1]) * np.cos(sph[2])
    y = sph[0] * np.sin(sph[1]) * np.sin(sph[2])
    z = sph[0] * sp.cos(sph[1])
    cart = np.squeeze(np.array((x,y,z)).T)
    return cart

def cart2sph(cart):
    """
    Convert one or more cartesian coordinates to spherical.

    Parameters
    ----------
    cart : shape 3 or Nx3 numpy.ndarray
        Cartesian coordinates of the form (x,y,z)

    Returns
    -------
    sph : shape 3 or Nx3 numpy.ndarray 
        Spherical coordinates of the form (r,theta,phi), where theta is the
        inclination angle from the Z axis and phi is the azimuth angle from the
        X axis.
    """
    cart = np.atleast_2d(cart).T
    r = np.sqrt(np.sum(cart**2,axis=0))
    th = np.arctan2( np.sqrt(np.sum(cart[:2]**2,axis=0)), cart[2] )
    ph = np.arctan2(cart[1],cart[0])
    cart = np.squeeze(np.array((r,th,ph)).T)
    return cart

def aoa_diff_rad(aoa_a,aoa_b):
    """
    Calculate the difference between two angles of arrival, in radians.
    """
    # Prepend radius 1 and convert to cartesian.
    cart_a = sph2cart( (1,) + tuple(aoa_a) )
    cart_b = sph2cart( (1,) + tuple(aoa_b) )
    # Property of dot product between vectors.
    return sp.arccos(sp.dot(cart_a,cart_b))

