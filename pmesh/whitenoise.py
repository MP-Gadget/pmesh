from . import _whitenoise
import numpy

def generate(complex, start, Nmesh, seed):
    _start = numpy.empty(complex.ndim, dtype='intp')
    _Nmesh = numpy.empty(complex.ndim, dtype='intp')
    _start[:] = start
    _Nmesh[:] = Nmesh

    _whitenoise.generate(complex, _start, _Nmesh, seed)
