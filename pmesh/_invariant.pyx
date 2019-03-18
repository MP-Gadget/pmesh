import cython
cimport cython
import numpy
cimport numpy
ctypedef cython.integral inttype
ctypedef cython.integral inttype2

numpy.import_array()

cdef extern from "_invariant_imp.c":
    ptrdiff_t pmesh_get_invariant_index(
        int ndim,
        const ptrdiff_t x[],
        const unsigned int cmask,
        const ptrdiff_t max_length) nogil;

def get_index(inttype [:, :] x, ptrdiff_t [:] Nmesh, compressed, maxlength=None):
    cdef ptrdiff_t [::1] xi
    cdef ptrdiff_t [::1] r
    cdef unsigned int cmask = 0
    cdef ptrdiff_t ml
    if maxlength is None:
        ml = -1
    else:
        ml = maxlength

    xi = numpy.empty(x.shape[1], dtype='intp')
    ra = numpy.empty(x.shape[0], dtype='intp')
    r = ra

    if compressed: # only compress the last axis
        cmask = 1 << (x.shape[1] - 1)
    cdef int bad
    with nogil:
        for i in range(x.shape[0]):
            bad = 0
            for d in range(x.shape[1]):
                xi[d] = x[i, d]
                # the convention in invariant fills positive first.
                # The nyquist frequency has a confusion -- it can be either.
                # we always map it to the positive.
                # PMesh and FFT uses a negative nyquist freq; correct for it here.
                if xi[d] == - Nmesh[d] // 2:
                    xi[d] = Nmesh[d] // 2

                if xi[d] > Nmesh[d] // 2 or xi[d] < -Nmesh[d] // 2:
                    bad = 1
            if bad:
                r[i] = -1
            else:
                r[i] = pmesh_get_invariant_index(x.shape[1], &xi[0], cmask, ml)

    return ra
