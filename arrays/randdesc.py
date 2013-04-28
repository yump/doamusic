import scipy as sp
import scipy.constants

# This antenna is an 18-element random array, with Z and Y coordinates from the
# file "randarray_inches.dat


# first column is Z axis.
# second column is inverted Y axis
zy = sp.loadtxt("randarray_inches.dat")
numel = 18
arr_in = sp.array((sp.zeros(numel),-zy[:,1],zy[:,0])).T

arr_m = 0.0254 * arr_in

sp.savetxt("randarray.dat", arr_m)
