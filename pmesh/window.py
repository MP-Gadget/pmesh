from ._window import Painter as PainterImp
import numpy
from numpy.lib.stride_tricks import as_strided

class PMPainter(PainterImp):
    def __cinit__(self, pm, support=2, window_type="linear"):
        PainterImp.__init__(self, pm, support, window_type)

    def paint(self, real, pos, mass=1):
        assert isinstance(real, numpy.ndarray)

        pos = numpy.asfarray(pos)
        mass = numpy.asarray(mass, dtype=pos.dtype)

        if len(mass.shape) == 0:
            mass = as_strided(mass, shape=pos.shape[:-1], strides=[0] * (len(pos.shape) - 1))

        PainterImp.paint(self, real, pos, mass)

    def readout(self, real, pos):
        assert isinstance(real, numpy.ndarray)

        pos = numpy.asfarray(pos)

        return PainterImp.readout(self, real, real.flat[0], pos)


