typedef struct PMeshWhiteNoiseGenerator {
    int ndim;
    unsigned int seed;
    ptrdiff_t Nmesh[32]; /* periodicity */
    ptrdiff_t start[32];

    void * canvas;
    int canvas_dtype_elsize;
    ptrdiff_t size[32];
    ptrdiff_t strides[32];
    unsigned int * seedtable[2][2];
} PMeshWhiteNoiseGenerator;

void
pmesh_whitenoise_generator_init(PMeshWhiteNoiseGenerator * generator);

void
pmesh_whitenoise_generator_fill(PMeshWhiteNoiseGenerator * generator);

