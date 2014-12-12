#coding=utf-8
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("distance_fun",["distance_fun.pyx"], include_dirs=[np.get_include()])]
)