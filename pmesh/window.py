from ._window import WindowResampler as _WindowResampler

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

class WindowResampler(_WindowResampler):
    def __init__(self, kind, support, ndim, scale=None, translate=None, period=None):
        kind = {
                'linear' : _WindowResampler.PAINTER_LINEAR,
                'lanczos' : _WindowResampler.PAINTER_LANCZOS,
               }[kind]

        if scale is None:
            scale = 1.0
        if translate is None:
            translate = 0
        if period is None:
            period = 0

        scale = _mkarr(scale, ndim, 'f8' )
        period = _mkarr(period, ndim, 'intp')
        translate = _mkarr(translate, ndim, 'intp')

        self.ndim = ndim
        self.scale = scale
        self.translate = translate
        self.period = period
        _WindowResampler.__init__(self, kind, support, ndim, scale, translate, period)

    def paint(self, real, pos, mass=None, diffdir=None):
        if diffdir is None: diffdir = -1
        else: diffdir %= self.ndim

        pos = numpy.asfarray(pos)
        if mass is None:
            mass = numpy.array(1.0, 'f8')
        else:
            mass = numpy.asfarray(mass)

        mass = _mkarr(mass, len(pos), mass.dtype)

        _WindowResampler.paint(self, real, pos, mass, diffdir)

    def readout(self, real, pos, out=None, diffdir=None):
        if diffdir is None: diffdir = -1
        else: diffdir %= self.ndim

        pos = numpy.asfarray(pos)
        if out is None:
            out = numpy.zeros(pos.shape[1:], dtype='f8')

        _WindowResampler.readout(self, real, pos, out, diffdir)

        return out
