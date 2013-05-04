#!/usr/bin/env python
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


from __future__ import absolute_import, division, print_function
import cProfile
import sys
from time import time
import itertools
import numpy as np
import scipy as sp
import scipy.misc
import scipy.constants
from scipy import pi

if __name__ == "__main__" and __package__ is None:
    sys.path.append('..')
    __package__ = "doamusic"
import doamusic
from doamusic import music
from doamusic import _music
from doamusic import util

# 16 element unit circle in the y-z plane
antx = sp.arange(16)
circarray = sp.array([0*antx, sp.cos(antx), sp.sin(antx)]).T

# 3 offset circles in planes paralell to y-z
front = circarray + [1,0,0]
back = circarray - [1,0,0]
triplecircarray = sp.concatenate((front,circarray,back))

# unit spacing grid (5x5)
gridarray = sp.array( 
    [ (0,y,z) for y,z in itertools.product(range(-3,4),repeat=2) ]
)

# unit spacing linear
linarray = sp.array( [ (0,y,0) for y in range(10) ] )

# Arrays as constructed.
wavelength = sp.constants.c/2.477e9

randarray = sp.loadtxt("arrays/randarray.dat")/wavelength
ourlinarray = sp.loadtxt("arrays/linarray.dat")/wavelength
ourcircarray = sp.loadtxt("arrays/circarray.dat")/wavelength

ants = randarray
nsamp = 21
snr = -6

s1_aoa = (pi/2,0)
#s2_aoa = (pi/2 + sp.randn()/2, sp.randn()/2)
s2_aoa = (pi/2+pi/6, -pi/6)
s1 = util.makesamples(ants,s1_aoa[0],s1_aoa[1],nsamp)
s2 = util.makesamples(ants,s2_aoa[0],s2_aoa[1],nsamp)

samples = s2 + s1
samples = util.awgn(samples,snr)

# add noise to s1 and s2
s1 = util.awgn(s1,snr)
s2 = util.awgn(s2,snr)

R = music.covar(samples)
est = music.Estimator(ants,R,nsignals=2)

def spectest(n=256):
    t = time()
    spec = est.spectrum((n,n))
    elapsed = time() - t
    print("spectrum calculation time: {}".format(elapsed))
    return spec

def sumspectest(dim=512,n=16):
    accum = sp.zeros((dim,dim))
    for i in range(n):
        mys1 = util.makesamples(ants,s1_aoa[0],s1_aoa[1],nsamp)
        mys2 = util.makesamples(ants,s2_aoa[0],s2_aoa[1],nsamp)
        mysamples = mys2 + mys1
        mysamples = util.awgn(mysamples,snr)
        cov = music.covar(mysamples)
        accum += music.Estimator(ants,cov,nsignals=2).spectrum((dim,dim))
    sp.misc.imsave("accumspec.png",accum/accum.max())

def doatest():
    print("s1 is {}".format(s1_aoa))
    print("s2 is {}".format(s2_aoa))
    s1_est = music.Estimator(ants,music.covar(s1),nsignals=1)
    s2_est = music.Estimator(ants,music.covar(s2),nsignals=1)
    # s1
    t1 = time()
    s1_res = s1_est.doasearch()[0]
    t1 = time() - t1
    s1_err = sp.rad2deg(util.aoa_diff_rad(s1_res,s1_aoa))
    print("s1: found {} in {}s, error {} deg".format(s1_res,t1,s1_err))
    # s2
    t2 = time()
    s2_res = s2_est.doasearch()[0]
    t2 = time() - t2
    s2_err = sp.rad2deg(util.aoa_diff_rad(s2_res,s2_aoa))
    print("s2: found {} in {}s, error {} deg".format(s2_res,t2,s2_err))
    # both signals
    bothres = est.doasearch()
    print("Both signals:\n{}".format(bothres))
    # timing

def cspec_error(n=64):
    specpy = est.spectrum((n,n),method=music._spectrum)
    specc = est.spectrum((n,n),method=_music.spectrum)
    sp.misc.imsave("c-spectrum.png",specc/np.max(specc))
    sp.misc.imsave("python-spectrum.png",specpy/np.max(specpy))
    return sp.mean(abs(specc-specpy))

def timetrial(reps=5):
    result = {}
    for i in range(5,10): # 32-512
        times = []
        for j in range(reps):
            t = time()
            _ = est.spectrum((2**i,2**i))
            times.append(time() - t)
        result[2**i] = min(times)
    return result

def indeptest(dim):
    R1 = music.covar(s1)
    R2 = music.covar(s2)
    s1spec = music.Estimator(ants,R1,nsignals=1).spectrum(dim)
    s2spec = music.Estimator(ants,R2,nsignals=1).spectrum(dim)
    bothspec = music.Estimator(ants,R,nsignals=2).spectrum(dim)
    sp.misc.imsave("s1spec.png",s1spec/s1spec.max())
    sp.misc.imsave("s2spec.png",s2spec/s2spec.max())
    sp.misc.imsave("bothspec.png",bothspec/bothspec.max())

def profile():
    cProfile.run("_ = est.spectrum((512,512))","spectrum.gprofile")
    cProfile.run("_ = doatest()","doasearch.gprofile")

if __name__ == '__main__':
    if sys.argv[1] == "profile":
        cProfile.run("_ = spectest(128)","spectrum.gprofile")
        cProfile.run("_ = doatest()","doasearch.gprofile")
    elif sys.argv[1] == "spectrum":
        if len(sys.argv) == 3:
            size = int(sys.argv[2])
        else:
            size = 512
        spec = spectest(size)
        sp.misc.imsave("spectrum.png",spec/np.max(spec))
        logspec = sp.log(spec/spec.min()) #positive only
        sp.misc.imsave("spectrum-log.png",logspec/logspec.max())
        # excluding the top 5%
        q95 = sp.sort(spec,axis=None)[-np.floor(0.05*(size**2))]
        lowspec = sp.clip(spec,0,q95)
        logq95 = sp.sort(spec,axis=None)[-np.floor(0.05*(size**2))]
        lowlogspec = sp.clip(logspec,0,logq95)
        sp.misc.imsave("spectrum-log-lows.png",lowlogspec)
        sp.misc.imsave("spectrum-lows.png",lowspec)
    elif sys.argv[1] == "check":
        print("Mean absolute deviation from python: {}".format(cspec_error()))
    elif sys.argv[1] == "timetrial":
        print("Times:")
        for i in timetrial().items():
            print("{}\t{}".format(*i))
    elif sys.argv[1] == "doasearch":
        indeptest((256,256))
        doatest()
    elif sys.argv[1] == "indep":
        indeptest((512,1024))
    elif sys.argv[1] == "sumspec":
        sumspectest(dim=512,n=int(sys.argv[2]))
    else:
        print("Bad arguments to _tests.py")
        exit(1)
    

