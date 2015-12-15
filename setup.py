from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import os

def myext(*args):
    return Extension(*args, include_dirs=["./", numpy.get_include()])

extensions = [
        myext("pmesh._domain", ["pmesh/_domain.pyx"]),
        ]

setup(
    name="pmesh", version="0.0.4",
    author="Yu Feng",
    description="Particle Mesh in Python",
    package_dir = {'pmesh': 'pmesh'},
    install_requires=['cython', 'numpy'],
    packages= ['pmesh'],
    requires=['numpy', 'mpi4py'],
    ext_modules = cythonize(extensions)
)

