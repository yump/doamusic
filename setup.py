from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

_music = Extension(
    name = "_music",
    sources = ["cmusic.c","_music.pyx"],
    include_dirs = ['cblas','m',numpy.get_include()],
    libraries = ["m","cblas"],
    extra_compile_args = ["-std=c99"])
                
setup(ext_modules=[_music],
      cmdclass = {'build_ext': build_ext})

