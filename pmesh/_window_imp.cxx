#define UNLIKELY(x) (x)
#define LIKELY(x) (x)

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "_window_imp.h"

template <typename FLOAT>
static void
_generic_paint(FastPMPainter * painter, double pos[], double weight);

template <typename FLOAT>
static double
_generic_readout(FastPMPainter * painter, double pos[]);

extern "C" {

#include "_window_wavelets.h"

static double
_linear_kernel(double x) {
    return 1.0 - fabs(x);
}

static double
_linear_diff(double x) {
    if( x < 0) {
        return 1;
    } else {
        return - 1;
    }
}

static double
_quadratic_kernel(double x) {
    /*
     * Take from https://arxiv.org/abs/0804.0070
     * */
    x = fabs(x);
    if(x <= 0.5) {
        return 0.75 - x * x;
    } else {
        x = 1.5 - x;
        return (x * x) * 0.5;
    }
}

static double
_quadratic_diff(double x) {
    double factor;
    if ( x < 0) {
        x = -x;
        factor = -1;
    } else {
        factor = +1;
    }

    if(x < 0.5) {
        return factor * (- 2 * x);
    } else {
        return factor * (- (1.5 - x));
    }
}

static double
_cubic_kernel(double x) {
    const double alpha = -0.5;
    /*
     * alpha = -0.5 is good. taken from
     * http://www.ipol.im/pub/art/2011/g_lmii/revisions/2011-09-27/g_lmii.html
     * */
    x = fabs(x);
    double xx = x * x;
    if(x < 1.0) {
        return (alpha + 2) * xx * x - (alpha + 3) * xx + 1;
    } else {
        return (alpha * xx * x) - 5 * alpha * xx + 8 * alpha * x - 4 * alpha;
    }
}

static double
_cubic_diff(double x) {
    const double alpha = -0.5;
    /*
     * alpha = -0.5 is good. taken from
     * http://www.ipol.im/pub/art/2011/g_lmii/revisions/2011-09-27/g_lmii.html
     * */
    double factor;
    if (x < 0) {
        factor = -1;
        x = -x;
    } else {
        factor = +1;
    }

    double xx = x * x;
    if(x < 1.0) {
        return factor * (3 * (alpha + 2) * xx - (alpha + 3));
    } else {
        return factor * (3 * (alpha * xx) - 10 * alpha * x + 8 * alpha);
    }
}

static inline double __cached__(int *status, double * table, double x, double (*func)(double)){
    const double dx = 1e-3;
    const double tablemax = dx * 16384;
    const double tablemin = dx * 1;
    if(!*status) {
        int i;
        for(i = 0; i < 16384; i ++) {
            double x = dx * i;
            table[i] = func(x);
        }
        *status = 1;
    }
    if(x > tablemin && x < tablemax) {
        int i = fabs(x) / dx;
        return table[i];
    }
    return func(x);
}


static double __sinc__(double x) {
    x *= 3.1415927;
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6. + x2  * x2 / 120.;
    } else {
        return sin(x) / x;
    }
}

static double __dsinc__(double x) {
    x *= 3.1415927;
    double r = 3.1415927;
    if(x < 1e-5 && x > -1e-5) {
        double xx = x * x;
        double xxxx = xx * xx;
        r *= - x / 3 + x*xx / 30 - xxxx*x/ 840 + xxxx * xx * x / 45360;
    } else {
        r *= 1 / x * cos(x) - 1 / (x *x) * sin(x);
    }
    return r;
}

/*
 * watch out these factors are incorrect -- probably due to caching.
 *
 * */
static double
_lanczos2_kernel(double x) {
    static int status = 0;
    static double table[16384];
    double s1 = __cached__(&status, table, x, __sinc__);
    double s2 = __cached__(&status, table, x * 0.5, __sinc__);
    return s1 * s2 / 1.00878984;
}

static double
_lanczos2_diff(double x) {
    static int status = 0;
    static double table[16384];

    double u1 = __cached__(&status, table, x, __sinc__);
    double u2 = __cached__(&status, table, x, __dsinc__);
    double v1 = __cached__(&status, table, x * 0.5, __sinc__);
    double v2 = __cached__(&status, table, x * 0.5, __dsinc__) * 0.5;
    return (u1 * v2 + u2 * v1) / 1.00878984;
}

static double
_lanczos3_kernel(double x) {
    static int status = 0;
    static double table[16384];
    double s1 = __cached__(&status, table, x, __sinc__);
    double s2 = __cached__(&status, table, x * 0.333333333333, __sinc__);
    return s1 * s2 / 0.997055;
}

static double
_lanczos3_diff(double x) {
    static int status = 0;
    static double table[16384];
    double u1 = __cached__(&status, table, x, __sinc__);
    double u2 = __cached__(&status, table, x, __dsinc__);
    double v1 = __cached__(&status, table, x * 0.333333333333, __sinc__);
    double v2 = __cached__(&status, table, x * 0.333333333333, __dsinc__) * 0.333333333333;
    return (u1 * v2 + u2 * v1) / 0.997055;
}

void
fastpm_painter_init(FastPMPainter * painter)
{
    if(painter->canvas_dtype_elsize == 8) {
        painter->paint = _generic_paint<double>;
        painter->readout = _generic_readout<double>;
    } else {
        painter->paint = _generic_paint<float>;
        painter->readout = _generic_readout<float>;
    }

    painter->left = (painter->support - 1) / 2;
    if (painter->support % 2 == 0){
        painter->shift = 0;
    } else {
        painter->shift = 0.5;
    }
    switch(painter->type) {
        case FASTPM_PAINTER_LINEAR:
            painter->kernel = _linear_kernel;
            painter->diff = _linear_diff;
            painter->nativesupport = 2;
        break;
        case FASTPM_PAINTER_QUADRATIC:
            painter->kernel = _quadratic_kernel;
            painter->diff = _quadratic_diff;
            painter->nativesupport = 3;
        break;
        case FASTPM_PAINTER_CUBIC:
            painter->kernel = _cubic_kernel;
            painter->diff = _cubic_diff;
            painter->nativesupport = 4;
        break;
        case FASTPM_PAINTER_LANCZOS2:
            painter->kernel = _lanczos2_kernel;
            painter->diff = _lanczos2_diff;
            painter->nativesupport = 4;
        break;
        case FASTPM_PAINTER_LANCZOS3:
            painter->kernel = _lanczos3_kernel;
            painter->diff = _lanczos3_diff;
            painter->nativesupport = 6;
        break;
        case FASTPM_PAINTER_DB12:
            painter->kernel = _db12_kernel;
            painter->diff = _db12_diff;
            painter->nativesupport = _db12_nativesupport;
        break;
        case FASTPM_PAINTER_DB20:
            painter->kernel = _db20_kernel;
            painter->diff = _db20_diff;
            painter->nativesupport = _db20_nativesupport;
        break;
    }
    int nmax = 1;
    int d;
    for(d = 0; d < painter->ndim; d++) {
        nmax *= (painter->support);
    }
    painter->Npoints = nmax;
    painter->vfactor = painter->nativesupport / (1. * painter->support);
}

void
fastpm_painter_paint(FastPMPainter * painter, double pos[], double weight)
{
    painter->paint(painter, pos, weight);
}

double
fastpm_painter_readout(FastPMPainter * painter, double pos[])
{
    return painter->readout(painter, pos);
}

static void
_fill_k(FastPMPainter * painter, double pos[], int ipos[], double k[][64])
{
    double gpos[painter->ndim];
    int d;

    for(d = 0; d < painter->ndim; d++) {
        gpos[d] = pos[d] * painter->scale[d];
        ipos[d] = floor(gpos[d] + painter->shift) - painter->left;
        double dx = gpos[d] - ipos[d]; /* relative to the left most nonzero.*/
        int i;
        double sum = 0;
        for(i = 0; i < painter->support; i ++) {
            double x = (dx - i) * painter->vfactor;
            k[d][i] = painter->kernel(x) * painter->vfactor;
            sum += k[d][i];

            /*
             * norm is still from the true kernel,
             * but we replace the value with the derivative
             * */
            if(d == painter->diffdir) {
                k[d][i] = painter->diff(x) * painter->scale[d] * painter->vfactor * painter->vfactor;
            }
        }
        /* normalize the kernel to conserve mass */
        for(i = 0; i < painter->support; i ++) {
            // k[d][i] /= sum;
        }
        ipos[d] += painter->translate[d];
    }
}

}

template <typename FLOAT>
static void
_generic_paint(FastPMPainter * painter, double pos[], double weight)
{
    int ipos[painter->ndim];
    /* the max support is 32 */
    double k[painter->ndim][64];

    char * canvas = (char*) painter->canvas;

    _fill_k(painter, pos, ipos, k);

    int rel[painter->ndim];
    for(int d =0; d < painter->ndim; d ++ ) rel[d] = 0;

    int s2 = painter->support;
    while(rel[0] != s2) {
        double kernel = 1.0;
        ptrdiff_t ind = 0;
        int d;
        for(d = 0; d < painter->ndim; d++) {
            int r = rel[d];
            int targetpos = ipos[d] + r;
            kernel *= k[d][r];
            if(painter->Nmesh[d] > 0) {
                while(targetpos >= painter->Nmesh[d]) {
                    targetpos -= painter->Nmesh[d];
                }
                while(targetpos < 0) {
                    targetpos += painter->Nmesh[d];
                }
            }
            if(UNLIKELY(targetpos >= painter->size[d]))
                goto outside;
            if(UNLIKELY(targetpos < 0))
                goto outside;
            ind += painter->strides[d] * targetpos;
        }
#pragma omp atomic
        * (FLOAT*) (canvas + ind) += weight * kernel;

    outside:
        rel[painter->ndim - 1] ++;
        for(d = painter->ndim - 1; d > 0; d --) {
            if(UNLIKELY(rel[d] == s2)) {
                rel[d - 1] ++;
                rel[d] = 0;
            }
        }
    }
    return;
}

template <typename FLOAT>
static double
_generic_readout(FastPMPainter * painter, double pos[])
{
    double value = 0;
    int ipos[painter->ndim];
    double k[painter->ndim][64];

    char * canvas = (char*) painter->canvas;

    _fill_k(painter, pos, ipos, k);

    int rel[painter->ndim];
    for(int d =0; d < painter->ndim; d++) rel[d] = 0;

    int s2 = painter->support;
    while(rel[0] != s2) {
        double kernel = 1.0;
        ptrdiff_t ind = 0;
        int d;
        for(d = 0; d < painter->ndim; d++) {
            int r = rel[d];

            kernel *= k[d][r];

            int targetpos = ipos[d] + r;

            if(painter->Nmesh[d] > 0) {
                while(targetpos >= painter->Nmesh[d]) {
                    targetpos -= painter->Nmesh[d];
                }
                while(targetpos < 0) {
                    targetpos += painter->Nmesh[d];
                }
            }
            if(UNLIKELY(targetpos >= painter->size[d])) {
                goto outside;
            }
            if(UNLIKELY(targetpos < 0))
                goto outside;
            ind += painter->strides[d] * targetpos;
        }
        value += kernel * *(FLOAT* )(canvas + ind);
outside:
        rel[painter->ndim - 1] ++;
        for(d = painter->ndim - 1; d > 0; d --) {
            if(UNLIKELY(rel[d] == s2)) {
                rel[d - 1] ++;
                rel[d] = 0;
            }
        }
    }
    return value;
}

