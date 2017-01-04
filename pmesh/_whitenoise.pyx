import cython
cimport cython
import numpy
cimport numpy
ctypedef cython.floating realtype

numpy.import_array()

cdef extern from "_whitenoise_imp.h":
    ctypedef struct PMeshWhiteNoiseGenerator:
        int ndim
        ptrdiff_t Nmesh[32]

        # determined during paint / readout
        void * canvas
        int canvas_dtype_elsize
        int unitary
        ptrdiff_t size[32]
        ptrdiff_t start[32]
        ptrdiff_t strides[32]
        unsigned int seed

    void pmesh_whitenoise_generator_init(PMeshWhiteNoiseGenerator * self)
    void pmesh_whitenoise_generator_fill(PMeshWhiteNoiseGenerator * self)

def generate(numpy.ndarray complex, ptrdiff_t [:] start, ptrdiff_t [:] Nmesh, unsigned int seed, int unitary):
    assert complex.dtype.kind == 'c'
    assert complex.ndim == 3

    cdef PMeshWhiteNoiseGenerator generator[1]

    generator.canvas = complex.data
    generator.canvas_dtype_elsize = complex.dtype.itemsize
    generator.ndim = complex.ndim
    for i in range(3):
        generator.size[i] =  complex.shape[i]
        generator.start[i] =  start[i]
        generator.strides[i] =  complex.strides[i]
        generator.Nmesh[i] =  Nmesh[i]

    generator.unitary = unitary
    generator.seed = seed

    pmesh_whitenoise_generator_init(generator)
    pmesh_whitenoise_generator_fill(generator)
