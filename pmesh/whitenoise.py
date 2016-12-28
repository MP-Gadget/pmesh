from . import _whitenoise
import numpy

def generate(complex, start, Nmesh, seed):
    _start = numpy.empty(complex.ndim, dtype='intp')
    _Nmesh = numpy.empty(complex.ndim, dtype='intp')
    _start[:] = start
    _Nmesh[:] = Nmesh

    if complex.ndim == 3:
        _whitenoise.generate(complex, _start, _Nmesh, seed)
    elif complex.ndim <= 2:
        # FIXME: this is not scale invariant though it is at least invariant
        # against partition. Since 2d and 1d is only used for testing this
        # may be good enough.
        rng = numpy.random.RandomState(seed)
        real = rng.normal(size=_Nmesh)
        full = numpy.fft.irfftn(real)
        full[...] *= numpy.prod(_Nmesh)
        slices = tuple([slice(a, a + b) for a, b in zip(_start, complex.shape)])
        complex[...] = full[slices]
    else:
        raise ValueError("Only knows how to make a whitenoise up to 3d")
