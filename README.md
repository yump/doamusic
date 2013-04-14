Doamusic
====

Purpose
----
Doamusic is a python library that implements the MUltiple SIgnal Classification
(or MUSIC) algorithm for direction of arrival estimation [as described by R.O.
Schmidt.](http://dx.doi.org/10.1109/TAP.1986.1143830).  

MUSIC, as applied to an N-element array, uses the NxN joint expectation matrix
(whose ij element is the expectation of the product of the signals at the ith
and jth elements) to discriminiate between up to N-1 uncorrelated incident
signals. Doamusic was developed for microwave direction finding, but it should
be usable for any array of coherent sensors.

Performance
----
Critical computations are multithreaded and using the BLAS in compiled native
code, and non-crtical computations use Numpy.  On 2006 hardware, Doamusic can
render a 512x512 pseudospectrum image for a 64 element array in around 2.3
seconds.
