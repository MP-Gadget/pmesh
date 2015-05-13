from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import os

def myext(*args):
    return Extension(*args, include_dirs=["./", numpy.get_include()])

extensions = [
        myext("pypm._domain", ["pypm/_domain.pyx"]),
        ]

setup(
    name="pypm", version="0.0",
    author="Yu Feng",
    description="Particle Mesh in Python",
    package_dir = {'pypm': 'pypm'},
    install_requires=['cython', 'numpy'],
    packages= ['pypm'],
    requires=['numpy', 'mpi4py'],
    ext_modules = cythonize(extensions)
)

