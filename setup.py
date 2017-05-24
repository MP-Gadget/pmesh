from distutils.core import setup
from distutils.extension import Extension
import numpy
import os

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")


extensions = [
        Extension("pmesh._domain", ["pmesh/_domain.pyx"], include_dirs=["./", numpy.get_include()]),
        Extension("pmesh._window", ["pmesh/_window.pyx", "pmesh/_window_imp.c"],
                depends=["pmesh/_window_imp.h", "pmesh/_window_generics_cic.h", "pmesh/_window_generics.h", "pmesh/_window_wavelets.h", "pmesh/_window_lanczos.h"], include_dirs=["./", numpy.get_include()]),
        Extension("pmesh._whitenoise", ["pmesh/gsl/ranlxd.c", "pmesh/gsl/missing.c", "pmesh/gsl/rng.c", "pmesh/_whitenoise_imp.c", "pmesh/_whitenoise.pyx"],
                depends=["pmesh/gsl/config.h", "pmesh/gsl/gsl_errno.h",
                         "pmesh/gsl/gsl_inline.h", "pmesh/gsl/gsl_rng.h",
                         "pmesh/gsl/gsl_types.h",
                         "pmesh/_whitenoise_imp.h", "pmesh/_whitenoise_generics.h"
                        ],
                include_dirs=["pmesh/gsl", "pmesh", numpy.get_include()])
        ]

from Cython.Build import cythonize
extensions = cythonize(extensions)

setup(
    name="pmesh", version=find_version("pmesh/version.py"),
    author="Yu Feng",
    description="Particle Mesh in Python",
    package_dir = {'pmesh': 'pmesh'},
    packages= ['pmesh', 'pmesh.tests'],
    install_requires=['cython', 'numpy', 'mpi4py', 'mpsort', 'pfft-python'],
    license='GPL3',
    ext_modules = extensions,
    extras_require={'full':['abopt'], 'abopt':['abopt']}
)

