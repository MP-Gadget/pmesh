import cython
cimport cython
import numpy
cimport numpy

ctypedef fused postype:
    cython.double
    cython.float

ctypedef fused masstype:
    cython.double
    cython.float

cdef extern from "_window_imp.h":

    ctypedef enum PMeshPainterType:
        PMESH_PAINTER_LINEAR
        PMESH_PAINTER_CUBIC
        PMESH_PAINTER_LANCZOS2
        PMESH_PAINTER_LANCZOS3
        PMESH_PAINTER_QUADRATIC
        PMESH_PAINTER_DB6
        PMESH_PAINTER_DB12
        PMESH_PAINTER_DB20
        PMESH_PAINTER_SYM6
        PMESH_PAINTER_SYM12
        PMESH_PAINTER_SYM20

    ctypedef struct PMeshPainter:
        PMeshPainterType type
        int support
        int ndim
        double scale[32]
        double translate[32]
        ptrdiff_t Nmesh[32]

        # determined during paint / readout
        int order[32]
        void * canvas
        int canvas_dtype_elsize
        ptrdiff_t size[32]
        ptrdiff_t strides[32]

    void pmesh_painter_init(PMeshPainter * painter)
    void pmesh_painter_paint(PMeshPainter * painter, double pos[], double mass)
    double pmesh_painter_readout(PMeshPainter * painter, double pos[])

cdef class ResampleWindow(object):
    cdef PMeshPainter painter[1]
    cdef readonly int support
    def __init__(self, kind, int support=-1):
        kinds = {
                'linear' : PMESH_PAINTER_LINEAR,
                'cubic' : PMESH_PAINTER_CUBIC,
                'quadratic' : PMESH_PAINTER_QUADRATIC,
                'lanczos2' : PMESH_PAINTER_LANCZOS2,
                'lanczos3' : PMESH_PAINTER_LANCZOS3,
                'db6' : PMESH_PAINTER_DB6,
                'db12' : PMESH_PAINTER_DB12,
                'db20' : PMESH_PAINTER_DB20,
                'sym6' : PMESH_PAINTER_SYM6,
                'sym12' : PMESH_PAINTER_SYM12,
                'sym20' : PMESH_PAINTER_SYM20,
               }

        cdef PMeshPainterType type

        if kind in kinds:
            type = <PMeshPainterType> <int> kinds[kind]
        else:
            type = <PMeshPainterType> <int> kind

        # FIXME: change this to scaling the size of the kernel
        self.painter.support = support
        self.painter.type = type
        self.painter.ndim = 0
        self.painter.canvas_dtype_elsize = 0

        pmesh_painter_init(self.painter)
        self.support = self.painter.support

    def paint(self, numpy.ndarray real, postype [:, :] pos, masstype [:] mass, order,
        double [:] scale, double [:] translate, ptrdiff_t [:] period):
        cdef double x[32]
        cdef double m
        cdef int d
        cdef int i

        assert real.dtype.kind == 'f'

        cdef PMeshPainter painter[1]

        painter[0] = self.painter[0]

        painter.ndim = real.ndim
        painter.canvas = <void*> real.data
        painter.canvas_dtype_elsize = real.dtype.itemsize

        for d in range(painter.ndim):
            painter.order[d] = order[d]
            painter.Nmesh[d] = period[d]
            painter.scale[d] = scale[d]
            painter.translate[d] = translate[d]

        for d in range(painter.ndim):
            painter.size[d] = real.shape[d]
            painter.strides[d] = real.strides[d]

        pmesh_painter_init(painter)

        for i in range(pos.shape[0]):
            for d in range(painter.ndim):
                x[d] = pos[i, d]
            m = mass[i]
            pmesh_painter_paint(painter, x, m)

    def readout(self, numpy.ndarray real, postype [:, :] pos, masstype [:] out, order,
        double [:] scale, double [:] translate, ptrdiff_t [:] period):

        cdef double x[32]
        cdef ptrdiff_t strides[32]
        cdef double m
        cdef int d
        cdef int i

        assert real.dtype.kind == 'f'

        cdef PMeshPainter painter[1]

        painter[0] = self.painter[0]

        painter.ndim = real.ndim
        painter.canvas = <void*> real.data
        painter.canvas_dtype_elsize = real.dtype.itemsize

        for d in range(painter.ndim):
            painter.order[d] = order[d]
            painter.Nmesh[d] = period[d]
            painter.scale[d] = scale[d]
            painter.translate[d] = translate[d]

        for d in range(painter.ndim):
            painter.size[d] = real.shape[d]
            painter.strides[d] = real.strides[d]

        pmesh_painter_init(painter)

        for i in range(pos.shape[0]):
            for d in range(painter.ndim):
                x[d] = pos[i, d]
            out[i] = pmesh_painter_readout(painter, x)

