import scipy as sp

# This is a 10 element linear array, with the closest spacing physically
# realizable with our antenna design.

# Unfortunately, the substrate is not perfectly rigid, so there may be some
# error in the x coordinates.

x = sp.zeros(10)
y = [
    0,
    -72,
    -144,
    -214,
    -286.5,
    -357,
    -427,
    -497.5,
    -569,
    -639.5
    ]
z = [
    -2,
    0,
    -1.5,
    -0.5,
    0,
    0,
    0,
    0,
    0,
    0
    ]

arr_mm = sp.array([x,y,z]).T

arr_m = arr_mm * 0.001 # Convert from millimeters to meters.

# Pad with zeros, because  we have an 18 channel receiver.
arr_padded = sp.concatenate((arr_m,sp.zeros((8,3))) )

sp.savetxt("linarray.dat",arr_padded)

