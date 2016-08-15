import cython
cimport cython
import numpy
cimport numpy

ctypedef cython.floating masstype
ctypedef cython.floating postype
ctypedef cython.floating realtype

cdef extern from "_window_imp.h":

    ctypedef enum FastPMPainterType:
        FASTPM_PAINTER_LINEAR
        FASTPM_PAINTER_CUBIC
        FASTPM_PAINTER_LANCZOS2
        FASTPM_PAINTER_LANCZOS3
        FASTPM_PAINTER_QUADRATIC
        FASTPM_PAINTER_DB6
        FASTPM_PAINTER_DB12
        FASTPM_PAINTER_DB20
        FASTPM_PAINTER_SYM6
        FASTPM_PAINTER_SYM12
        FASTPM_PAINTER_SYM20

    ctypedef struct FastPMPainter:
        FastPMPainterType type
        int support
        int ndim
        double scale[32]
        double translate[32]
        ptrdiff_t Nmesh[32]

        # determined during paint / readout
        int diffdir
        void * canvas
        int canvas_dtype_elsize
        ptrdiff_t size[32]
        ptrdiff_t strides[32]

    void fastpm_painter_init(FastPMPainter * painter)
    void fastpm_painter_paint(FastPMPainter * painter, double pos[], double mass)
    double fastpm_painter_readout(FastPMPainter * painter, double pos[])

cdef class ResampleWindow(object):
    cdef FastPMPainter painter[1]
    cdef readonly int support
    def __init__(self, kind, int support=-1):
        kinds = {
                'linear' : FASTPM_PAINTER_LINEAR,
                'cubic' : FASTPM_PAINTER_CUBIC,
                'quadratic' : FASTPM_PAINTER_QUADRATIC,
                'lanczos2' : FASTPM_PAINTER_LANCZOS2,
                'lanczos3' : FASTPM_PAINTER_LANCZOS3,
                'db6' : FASTPM_PAINTER_DB6,
                'db12' : FASTPM_PAINTER_DB12,
                'db20' : FASTPM_PAINTER_DB20,
                'sym6' : FASTPM_PAINTER_SYM6,
                'sym12' : FASTPM_PAINTER_SYM12,
                'sym20' : FASTPM_PAINTER_SYM20,
               }

        cdef FastPMPainterType type

        if kind in kinds:
            type = <FastPMPainterType> <int> kinds[kind]
        else:
            type = <FastPMPainterType> <int> kind

        # FIXME: change this to scaling the size of the kernel
        self.painter.support = support
        self.painter.type = type
        self.painter.ndim = 0
        self.painter.canvas_dtype_elsize = 0

        fastpm_painter_init(self.painter)
        self.support = self.painter.support

    def paint(self, numpy.ndarray real, postype [:, :] pos, masstype [:] mass, int diffdir,
        double [:] scale, double [:] translate, ptrdiff_t [:] period):
        cdef double x[32]
        cdef double m
        cdef int d
        cdef int i

        assert real.dtype.kind == 'f'

        cdef FastPMPainter painter[1]

        painter[0] = self.painter[0]

        painter.ndim = real.ndim
        painter.canvas = <void*> real.data
        painter.canvas_dtype_elsize = real.dtype.itemsize
        painter.diffdir = diffdir

        for d in range(painter.ndim):
            painter.Nmesh[d] = period[d]
            painter.scale[d] = scale[d]
            painter.translate[d] = translate[d]

        for d in range(painter.ndim):
            painter.size[d] = real.shape[d]
            painter.strides[d] = real.strides[d]

        fastpm_painter_init(painter)

        for i in range(pos.shape[0]):
            for d in range(painter.ndim):
                x[d] = pos[i, d]
            m = mass[i]
            fastpm_painter_paint(painter, x, m)

    def readout(self, numpy.ndarray real, postype [:, :] pos, masstype [:] out, int diffdir,
        double [:] scale, double [:] translate, ptrdiff_t [:] period):

        cdef double x[32]
        cdef ptrdiff_t strides[32]
        cdef double m
        cdef int d
        cdef int i

        assert real.dtype.kind == 'f'

        cdef FastPMPainter painter[1]

        painter[0] = self.painter[0]

        painter.ndim = real.ndim
        painter.canvas = <void*> real.data
        painter.canvas_dtype_elsize = real.dtype.itemsize
        painter.diffdir = diffdir

        for d in range(painter.ndim):
            painter.Nmesh[d] = period[d]
            painter.scale[d] = scale[d]
            painter.translate[d] = translate[d]

        for d in range(painter.ndim):
            painter.size[d] = real.shape[d]
            painter.strides[d] = real.strides[d]

        fastpm_painter_init(painter)

        for i in range(pos.shape[0]):
            for d in range(painter.ndim):
                x[d] = pos[i, d]
            out[i] = fastpm_painter_readout(painter, x)

