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

ctypedef fused hsmltype:
    cython.double
    cython.float

cdef extern from "_window_imp.h":

    ctypedef enum PMeshPainterType:
        PMESH_PAINTER_NEAREST
        PMESH_PAINTER_LINEAR
        PMESH_PAINTER_CUBIC
        PMESH_PAINTER_LANCZOS2
        PMESH_PAINTER_LANCZOS3
        PMESH_PAINTER_LANCZOS4
        PMESH_PAINTER_LANCZOS5
        PMESH_PAINTER_LANCZOS6
        PMESH_PAINTER_ACG2
        PMESH_PAINTER_ACG3
        PMESH_PAINTER_ACG4
        PMESH_PAINTER_ACG5
        PMESH_PAINTER_ACG6
        PMESH_PAINTER_QUADRATIC
        PMESH_PAINTER_DB6
        PMESH_PAINTER_DB12
        PMESH_PAINTER_DB20
        PMESH_PAINTER_SYM6
        PMESH_PAINTER_SYM12
        PMESH_PAINTER_SYM20
        PMESH_PAINTER_TUNED_NNB
        PMESH_PAINTER_TUNED_CIC
        PMESH_PAINTER_TUNED_TSC
        PMESH_PAINTER_TUNED_PCS

    ctypedef struct PMeshPainter:
        PMeshPainterType type
        int support
        int nativesupport
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
    void pmesh_painter_paint(PMeshPainter * painter, double pos[], double mass, double hsml)
    double pmesh_painter_readout(PMeshPainter * painter, double pos[], double hsml)
    double pmesh_painter_get_fwindow(PMeshPainter * painter, double w)

cdef class ResampleWindow(object):
    cdef PMeshPainter painter[1]
    cdef readonly int nativesupport
    cdef readonly int support
    def __init__(self, kind, int support=-1):
        kinds = {
                'tunednnb' : PMESH_PAINTER_TUNED_NNB,
                'tunedcic' : PMESH_PAINTER_TUNED_CIC,
                'tunedtsc' : PMESH_PAINTER_TUNED_TSC,
                'tunedpcs' : PMESH_PAINTER_TUNED_PCS,
                'nearest' : PMESH_PAINTER_NEAREST,
                'linear' : PMESH_PAINTER_LINEAR,
                'quadratic' : PMESH_PAINTER_QUADRATIC,
                'cubic' : PMESH_PAINTER_CUBIC,
                'lanczos2' : PMESH_PAINTER_LANCZOS2,
                'lanczos3' : PMESH_PAINTER_LANCZOS3,
                'lanczos4' : PMESH_PAINTER_LANCZOS4,
                'lanczos5' : PMESH_PAINTER_LANCZOS5,
                'lanczos6' : PMESH_PAINTER_LANCZOS6,
                'acg2' : PMESH_PAINTER_ACG2,
                'acg3' : PMESH_PAINTER_ACG3,
                'acg4' : PMESH_PAINTER_ACG4,
                'acg5' : PMESH_PAINTER_ACG5,
                'acg6' : PMESH_PAINTER_ACG6,
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

        self.nativesupport = self.painter.nativesupport
        self.support = self.painter.support
        self.kind = kind

    def get_fwindow(self, double [:] w):
        cdef double [:] vrt
        cdef double v
        rt = numpy.zeros(w.shape[0], dtype='f8')
        vrt = rt
        for i in range(w.shape[0]):
            v = pmesh_painter_get_fwindow(self.painter, w[i])
            vrt[i] = v
        return rt

    def paint(self, numpy.ndarray real, postype [:, :] pos, hsmltype [:] hsml, masstype [:] mass,
            order, double [:] scale, double [:] translate, ptrdiff_t [:] period):
        cdef double x[32]
        cdef double m, h
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
            if hsml is not None:
                h = hsml[i]
            else:
                h = 1.0
            pmesh_painter_paint(painter, x, m, h)

    def readout(self, numpy.ndarray real, postype [:, :] pos, hsmltype [:] hsml, masstype [:] out, order,
        double [:] scale, double [:] translate, ptrdiff_t [:] period):

        cdef double x[32]
        cdef ptrdiff_t strides[32]
        cdef double m, h
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
            if hsml is not None:
                h = hsml[i]
            else:
                h = 1.0
            out[i] = pmesh_painter_readout(painter, x, h)

