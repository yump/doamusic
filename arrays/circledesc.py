import scipy as sp
import scipy.constants

# This antenna is an 18-element circular array, with radius 12.5 in
# operating at 2.4771 GHz

numel = 18

radius = 0.0254*12.5

arr = [ 
        (0,radius*sp.cos(th), radius*sp.sin(th)) 
        for th in sp.linspace(0, 2*sp.pi, numel, endpoint=False) 
      ]

sp.savetxt("circarray.dat", sp.array(arr))
