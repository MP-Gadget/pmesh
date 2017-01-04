from . import _whitenoise
import numpy

def generate(complex, start, Nmesh, seed, unitary):
    """
        unitary : bool
            True for a unitary gaussian field (amplitude is fixed to 1)
            False for a true gaussian field
    """
    _start = numpy.empty(complex.ndim, dtype='intp')
    _Nmesh = numpy.empty(complex.ndim, dtype='intp')
    _start[:] = start
    _Nmesh[:] = Nmesh

    if complex.ndim == 3:
        _whitenoise.generate(complex, _start, _Nmesh, seed, unitary)
    elif complex.ndim <= 2:
        # FIXME: this is not scale invariant though it is at least invariant
        # against partition. Since 2d and 1d is only used for testing this
        # may be good enough.
        rng = numpy.random.RandomState(seed)
        real = rng.normal(size=_Nmesh)
        full = numpy.fft.rfftn(real)
        full[...] *= numpy.prod(_Nmesh) ** -0.5
        slices = tuple([slice(a, a + b) for a, b in zip(_start, complex.shape)])
        complex[...] = full[slices]
        if unitary:
            # there is a problem when the amplitude is too small,
            # but because we do not use 1d and 2d for anything serious
            # it is probably OK assuming they just have phase of zero.
            complex[...] = numpy.exp(1j * numpy.angle(complex))
    else:
        raise ValueError("Only knows how to make a whitenoise up to 3d")
