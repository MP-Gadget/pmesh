import cython
cimport cython
import numpy
cimport numpy

ctypedef cython.floating masstype
ctypedef cython.floating postype
ctypedef cython.floating realtype

cdef extern from "_window_imp.h":
    ctypedef struct PMInterface:
        int ndim
        ptrdiff_t Nmesh[32]
        double BoxSize[32]

        ptrdiff_t start[32]
        ptrdiff_t size[32]
        ptrdiff_t strides[32]

    struct D:
        void   (*paint)(FastPMPainter * painter, double * canvas, ptrdiff_t strides[], double pos[], double weight, int diffdir) nogil
        double (*readout)(FastPMPainter * painter, double * canvas, ptrdiff_t strides[], double pos[], int diffdir) nogil
    struct S:
        void   (*paint)(FastPMPainter * painter, float * canvas, ptrdiff_t strides[], double pos[], double weight, int diffdir) nogil
        double (*readout)(FastPMPainter * painter, float * canvas, ptrdiff_t strides[], double pos[], int diffdir) nogil
    ctypedef struct FastPMPainter:
        D DOUBLE
        S SINGLE
        int support
        int diffdir

    ctypedef enum FastPMPainterType:
        FASTPM_PAINTER_LINEAR
        FASTPM_PAINTER_LANCZOS

    void fastpm_painter_init(FastPMPainter * painter, PMInterface * pm,
            FastPMPainterType type, int support)

    void fastpm_painter_init_diff(FastPMPainter * painter, FastPMPainter * base, int diffdir)

cdef class Painter(object):
    cdef FastPMPainter painter[1]
    cdef PMInterface pmi[1]
    cdef readonly object pm
    property support:
        def __get__(self):
            return self.painter.support
    cdef readonly int ndim

    def __cinit__(self, pm, support=2, window_type="linear", diffdir=None):
        """ create a painter.

            Parameters
            ----------
            window_type : "linear", "lanczos"

        """
        self.pmi.ndim = len(pm.Nmesh)
        self.ndim = self.pmi.ndim
        for d in range(self.pmi.ndim):
            self.pmi.Nmesh[d] = pm.Nmesh[d]
            self.pmi.BoxSize[d] = pm.BoxSize[d]
            self.pmi.start[d] = pm.partition.local_i_start[d]
            self.pmi.size[d] = pm.partition.local_ni[d]

        type = {
                'linear' : FASTPM_PAINTER_LINEAR,
                'lanczos' : FASTPM_PAINTER_LANCZOS,
               }[window_type]

        fastpm_painter_init(self.painter, self.pmi, type, support)
        if diffdir is not None:
            diffdir %= len(self.pm.Nmesh)
            fastpm_painter_init_diff(self.painter, self.painter, diffdir)

        self.pm = pm

    def paint(self, numpy.ndarray real, postype [:, :] pos, masstype [:] mass):
        cdef double x[32]
        cdef ptrdiff_t strides[32]
        cdef double m
        cdef int d
        cdef int i

        assert real.dtype.kind == 'f'

        for d in range(self.ndim):
            strides[d] = real.strides[d] / real.dtype.itemsize

        for i in range(pos.shape[0]):
            for d in range(self.ndim):
                x[d] = pos[i, d]
            m = mass[i]
            if real.dtype.itemsize == 8:
                self.painter.DOUBLE.paint(self.painter, <double*> (real.data), strides, x, m, self.painter.diffdir)
            if real.dtype.itemsize == 4:
                self.painter.SINGLE.paint(self.painter, <float*> (real.data), strides, x, m, self.painter.diffdir)

    def readout(self, numpy.ndarray real, postype [:, :] pos, masstype [:] out):
        cdef double x[32]
        cdef ptrdiff_t strides[32]
        cdef double m
        cdef int d
        cdef int i

        assert real.dtype.kind == 'f'

        for d in range(self.ndim):
            strides[d] = real.strides[d] / real.dtype.itemsize

        for i in range(pos.shape[0]):
            for d in range(self.ndim):
                x[d] = pos[i, d]
                if real.dtype.itemsize == 8:
                    result = self.painter.DOUBLE.readout(self.painter, <double*> (real.data), strides, x, self.painter.diffdir)
                if real.dtype.itemsize == 4:
                    result = self.painter.SINGLE.readout(self.painter, <float*> (real.data), strides, x, self.painter.diffdir)
                out[i] = result
