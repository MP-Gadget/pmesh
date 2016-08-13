from ._window import ResampleWindow as _ResampleWindow

import numpy
from numpy.lib.stride_tricks import as_strided

def _mkarr(var, shape, dtype):
    var = numpy.asarray(var, dtype=dtype)
    if numpy.isscalar(shape):
        shape = (int(shape),)

    if len(var.shape) == 0:
        return as_strided(var, shape=shape, strides=[0] * len(shape))
    else:
        r = numpy.empty(shape, dtype)
        r[...] = var
        return r

class Affine(object):
    """ Defines an affine Transformation, used by ResampleWindow.
    """
    def __init__(self, ndim, scale=None, translate=None, period=None):
        if scale is None:
            scale = 1.0
        if translate is None:
            translate = 0
        if period is None:
            period = 0

        scale = _mkarr(scale, ndim, 'f8' )
        period = _mkarr(period, ndim, 'intp')
        translate = _mkarr(translate, ndim, 'intp')

        self.scale = scale
        self.translate = translate
        self.period = period

class ResampleWindow(_ResampleWindow):
    def __init__(self, kind, support):
        _ResampleWindow.__init__(self, kind, support)

    def paint(self, real, pos, mass=None, diffdir=None, transform=None):
        if transform is None:
            transform = Affine(real.ndim)

        assert isinstance(transform, Affine)

        if diffdir is None: diffdir = -1
        else: diffdir %= real.ndim

        pos = numpy.asfarray(pos)
        if mass is None:
            mass = numpy.array(1.0, 'f8')
        else:
            mass = numpy.asfarray(mass)

        mass = _mkarr(mass, len(pos), mass.dtype)


        _ResampleWindow.paint(self, real, pos, mass, diffdir, transform.scale, transform.translate, transform.period)

    def readout(self, real, pos, out=None, diffdir=None, transform=None):
        if transform is None:
            transform = Affine(real.ndim)

        assert isinstance(transform, Affine)

        if diffdir is None: diffdir = -1
        else: diffdir %= real.ndim

        pos = numpy.asfarray(pos)
        if out is None:
            out = numpy.zeros(pos.shape[:-1], dtype='f8')

        _ResampleWindow.readout(self, real, pos, out, diffdir, transform.scale, transform.translate, transform.period)

        return out

CIC = ResampleWindow(kind="linear", support=2)
TSC = ResampleWindow(kind="quadratic", support=3)
CUBIC = ResampleWindow(kind="cubic", support=4)
LANCZOS2 = ResampleWindow(kind="lanczos2", support=4)
LANCZOS3 = ResampleWindow(kind="lanczos3", support=6)
DB12 = ResampleWindow(kind="db12", support=10)
DB20 = ResampleWindow(kind="db20", support=13)

