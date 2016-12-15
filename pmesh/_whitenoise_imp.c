#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <gsl/config.h>
#include <gsl/gsl_rng.h>
#include "_whitenoise_imp.h"

void
pmesh_whitenoise_generator_init(PMeshWhiteNoiseGenerator * generator)
{
    if(generator->ndim != 3) {
        abort();
    }
    /* do nothing */
}

static inline void 
SETSEED(PMeshWhiteNoiseGenerator * self, int i, int j, gsl_rng * rng) 
{ 
    unsigned int seed = 0x7fffffff * gsl_rng_uniform(rng); 

    int ii[2] = {i, self->Nmesh[0] - i};
    int jj[2] = {j, self->Nmesh[1] - j};
    int d1, d2;
    for(d1 = 0; d1 < 2; d1++) {
        ii[d1] -= self->start[0];
        jj[d1] -= self->start[1];
    }
    for(d1 = 0; d1 < 2; d1++)
    for(d2 = 0; d2 < 2; d2++) {
        if( ii[d1] >= 0 && 
            ii[d1] < self->size[0] &&
            jj[d2] >= 0 &&
            jj[d2] < self->size[1]
        ) {
            self->seedtable[d1][d2][ii[d1] * self->size[1] + jj[d2]] = seed;
        }
    }
}

static inline unsigned int 
GETSEED(PMeshWhiteNoiseGenerator * self, int i, int j, int d1, int d2) 
{
    i -= self->start[0];
    j -= self->start[1];

    /* these shall never happen: */
    if(i < 0) abort();
    if(j < 0) abort();
    if(i >= self->size[0]) abort();
    if(j >= self->size[1]) abort();

    return self->seedtable[d1][d2][i * self->size[1] + j];
}

#define FLOAT float
#define mkname(a) a ## _ ## float
#include "_whitenoise_generics.h"
#undef FLOAT
#undef mkname
#define mkname(a) a ## _ ## double
#define FLOAT double
#include "_whitenoise_generics.h"
#undef FLOAT
#undef mkname

void
pmesh_whitenoise_generator_fill(PMeshWhiteNoiseGenerator * self)
{
    /* store the seed of all possible hermitian conjugate modes,
     * for some axis are self conjugate we blindly save the negative mode for each
     * quadrant */

    int i, j;
    for(i = 0; i < 2; i ++)
    for(j = 0; j < 2; j ++) {
        self->seedtable[i][j] = calloc(self->size[0] * self->size[1], sizeof(int));
    }

    if(self->canvas_dtype_elsize == 16) {
        /* complex128*/
        _generic_fill_double(self, (double *) self->canvas, self->seed);
    } else {
        /* complex64*/
        _generic_fill_float(self, (float *) self->canvas, self->seed);
    }

    for(i = 0; i < 2; i ++)
    for(j = 0; j < 2; j ++) {
        free(self->seedtable[i][j]);
        self->seedtable[i][j] = NULL;
    }
}

