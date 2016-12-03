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

        Parameters
        ----------
            translate : array_like, in integer mesh units.
            period : array_like in integer mesh units.
            scale : factor that multiples on position to obtain mesh units.

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
        translate = _mkarr(translate, ndim, 'f8')

        self.scale = scale
        self.translate = translate
        self.period = period
        self.ndim = ndim

    def shift(self, amount):
        """ Returns a new Affine where the translate is shifted by amount.
            Amount is in integer mesh units, as translate. """

        return Affine(self.ndim, self.scale, self.translate + amount, self.period)

class ResampleWindow(_ResampleWindow):
    def __init__(self, kind, support=-1):
        _ResampleWindow.__init__(self, kind, support)

    def paint(self, real, pos, mass=None, diffdir=None, transform=None):
        """
            paint to a field.

            Parameters
            ----------
            real : array_like
                original values are preserved.

            pos : array_like

            mass : array_like or None
                None for 1

            diffdir: int or None
                direction for differentiation kernel.
                0, 1, 2,... or None

            transform: Affine
                The Affine transformation from position to grid units.

        """
        if transform is None:
            transform = Affine(real.ndim)

        assert isinstance(transform, Affine)

        order = numpy.zeros(real.ndim, dtype=int)
        if diffdir is not None:
            order[diffdir] = 1

        pos = numpy.asfarray(pos)
        if mass is None:
            mass = numpy.array(1.0, 'f8')
        else:
            mass = numpy.asfarray(mass)

        mass = _mkarr(mass, len(pos), mass.dtype)

        _ResampleWindow.paint(self, real, pos, mass, order, transform.scale, transform.translate, transform.period)

    def readout(self, real, pos, out=None, diffdir=None, transform=None):
        """
            readout from a field.

            Parameters
            ----------
            real : array_like

            pos : array_like

            out : array_like

            mass : array_like or None
                None for 1

            diffdir: int or None
                direction for differentiation kernel.
                0, 1, 2,... or None

            transform: Affine
                The Affine transformation from position to grid units.

        """
        if transform is None:
            transform = Affine(real.ndim)

        assert isinstance(transform, Affine)

        order = numpy.zeros(real.ndim, dtype=int)
        if diffdir is not None:
            order[diffdir] = 1

        pos = numpy.asfarray(pos)
        if out is None:
            out = numpy.zeros(pos.shape[:-1], dtype='f8')

        _ResampleWindow.readout(self, real, pos, out, order, transform.scale, transform.translate, transform.period)

        return out

methods = dict(
    CIC = ResampleWindow(kind="linear"),
    TSC = ResampleWindow(kind="quadratic"),
    CUBIC = ResampleWindow(kind="cubic"),
    LANCZOS2 = ResampleWindow(kind="lanczos2"),
    LANCZOS3 = ResampleWindow(kind="lanczos3"),
    DB6 = ResampleWindow(kind="db6"),
    DB12 = ResampleWindow(kind="db12"),
    DB20 = ResampleWindow(kind="db20"),
    SYM6 = ResampleWindow(kind="sym6"),
    SYM12 = ResampleWindow(kind="sym12"),
    SYM20 = ResampleWindow(kind="sym20"),
)
for m, p in list(methods.items()):
    methods[m.lower()] = p
    globals()[m] = p

del m, p
