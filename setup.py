from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import mpi4py
import os
compiler = mpi4py.get_config()['mpicc']
os.environ['CC'] = compiler
os.environ['LDSHARED'] = compiler + ' -shared'

def myext(*args):
    return Extension(*args, include_dirs=["./", numpy.get_include()])

extensions = [
        myext("pypm._domain", ["src/_domain.pyx"]),
        ]

setup(
    name="pypm", version="0.0",
    author="Yu Feng",
    description="Particle Mesh in Python",
    package_dir = {'pypm': 'src'},
    install_requires=['cython', 'numpy', 'mpi4py'],
    packages= ['pypm'],
    requires=['numpy', 'mpi4py'],
    ext_modules = cythonize(extensions)
)

