from __future__ import absolute_import, division, print_function
import music
import scipy as sp
from scipy import pi
import scipy.misc
import util
from time import time

# 16 element unit circle in the y-z plane
antx = sp.arange(16)
circarray = sp.array([0*antx, sp.cos(antx), sp.sin(antx)]).T

# 3 offset circles in planes paralell to y-z
front = circarray + [1,0,0]
back = circarray - [1,0,0]
triplecircarray = sp.concatenate((front,circarray,back))

ants = circarray
nsamp = 32
snr = 20

s1 = util.makesamples(ants,pi/2,0,nsamp,snr)
s2 = util.makesamples(ants,pi/2 + sp.rand()-.5,sp.randn()-.5,nsamp,snr)

samples = s2

R = music.covar(samples)
est = music.Estimator(ants,R,nsignals=2)

def spectest(n=64):
	t = time()
	spec = music.spectrum(est,sp.linspace(0,pi,n),sp.linspace(0,2*pi,n))
	elapsed = time() - t
	print("spectrum calculation time: {}".format(elapsed))
	sp.misc.imsave("music-spectrum.png",spec)
	return spec


def doatest():
	raise NotImplementedError()

