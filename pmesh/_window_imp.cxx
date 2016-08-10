#define UNLIKELY(x) (x)
#define LIKELY(x) (x)

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "_window_imp.h"

template <typename FLOAT>
static void
_generic_paint(FastPMPainter * painter, FLOAT * canvas, ptrdiff_t strides[], double pos[], double weight, int diffdir);

template <typename FLOAT>
static double
_generic_readout(FastPMPainter * painter, FLOAT * canvas, ptrdiff_t strides[], double pos[], int diffdir);

extern "C" {

static double
_linear_kernel(double x, double invh) {
    return 1.0 - fabs(x * invh);
}

static double
_linear_diff(double x, double invh) {
    if( x < 0) {
        return 1 * invh;
    } else {
        return - 1 * invh;
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

static double
_lanczos_kernel(double x, double invh) {
    static int status = 0;
    static double table[16384];
    double s1 = __cached__(&status, table, x, __sinc__);
    double s2 = __cached__(&status, table, x * invh, __sinc__);
    return s1 * s2;
}

static double
_lanczos_diff(double x, double invh) {
    static int status = 0;
    static double table[16384];
    double u1 = __cached__(&status, table, x, __sinc__);
    double u2 = __cached__(&status, table, x, __dsinc__);
    double v1 = __cached__(&status, table, x * invh, __sinc__);
    double v2 = __cached__(&status, table, x * invh, __dsinc__) * invh;
    return u1 * v2 + u2 * v1;
}

void
fastpm_painter_init(FastPMPainter * painter, PMInterface * pm,
    FastPMPainterType type, int support)
{
    painter->pm = pm;
    painter->DOUBLE.paint = _generic_paint<double>;
    painter->DOUBLE.readout = _generic_readout<double>;
    painter->SINGLE.paint = _generic_paint<float>;
    painter->SINGLE.readout = _generic_readout<float>;

    painter->support = support;
    painter->hsupport = 0.5 * support;
    painter->invh= 1 / (0.5 * support);
    painter->left = (support  - 1) / 2;
    painter->diffdir = -1;

    switch(type) {
        case FASTPM_PAINTER_LINEAR:
            painter->kernel = _linear_kernel;
            painter->diff = _linear_diff;
        break;
        case FASTPM_PAINTER_LANCZOS:
            painter->kernel = _lanczos_kernel;
            painter->diff = _lanczos_diff;
        break;
    }
    int nmax = 1;
    int d;
    for(d = 0; d < pm->ndim; d++) {
        nmax *= (support);
        painter->InvCellSize[d] = pm->Nmesh[d] / pm->BoxSize[d];
    }
    painter->Npoints = nmax;
}

void
fastpm_painter_init_diff(FastPMPainter * painter, FastPMPainter * base, int diffdir)
{
    *painter = *base;
    painter->diffdir = diffdir;
}

static void
_fill_k(FastPMPainter * painter, double pos[], int ipos[], double k[][64], int diffdir)
{
    PMInterface * pm = painter->pm;
    double gpos[pm->ndim];
    int d;
    for(d = 0; d < pm->ndim; d++) {
        gpos[d] = pos[d] * painter->InvCellSize[d];
        ipos[d] = floor(gpos[d]) - painter->left;
        double dx = gpos[d] - ipos[d];
        int i;
        double sum = 0;
        for(i = 0; i < painter->support; i ++) {
            k[d][i] = painter->kernel(dx - i, painter->invh);
            sum += k[d][i];

            /*
             * norm is still from the true kernel,
             * but we replace the value with the derivative
             * */
            if(d == diffdir) {
                k[d][i] = painter->diff(dx - i, painter->invh) * painter->InvCellSize[d];
            }
        }
        /* normalize the kernel to conserve mass */
        for(i = 0; i < painter->support; i ++) {
            k[d][i] /= sum;
        }
        ipos[d] -= pm->start[d];
    }
}

}

template <typename FLOAT>
static void
_generic_paint(FastPMPainter * painter, FLOAT * canvas, ptrdiff_t strides[], double pos[], double weight, int diffdir)
{
    PMInterface * pm = painter->pm;
    int ipos[pm->ndim];
    /* the max support is 32 */
    double k[pm->ndim][64];

    _fill_k(painter, pos, ipos, k, diffdir);

    int rel[pm->ndim] = {0};
    int s2 = painter->support;
    while(rel[0] != s2) {
        double kernel = 1.0;
        ptrdiff_t ind = 0;
        int d;
        for(d = 0; d < pm->ndim; d++) {
            int r = rel[d];
            int targetpos = ipos[d] + r;
            kernel *= k[d][r];
            while(targetpos >= pm->Nmesh[d]) {
                targetpos -= pm->Nmesh[d];
            }
            while(targetpos < 0) {
                targetpos += pm->Nmesh[d];
            }
            ind += strides[d] * targetpos;
            if(UNLIKELY(targetpos >= pm->size[d]))
                goto outside;
        }
#pragma omp atomic
        canvas[ind] += weight * kernel;

    outside:
        rel[pm->ndim - 1] ++;
        for(d = pm->ndim - 1; d > 0; d --) {
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
_generic_readout(FastPMPainter * painter, FLOAT * canvas, ptrdiff_t strides[], double pos[], int diffdir)
{
    PMInterface * pm = painter->pm;
    double value = 0;
    int ipos[pm->ndim];
    double k[pm->ndim][64];

    _fill_k(painter, pos, ipos, k, diffdir);

    int rel[pm->ndim] = {0};

    int s2 = painter->support;
    while(rel[0] != s2) {
        double kernel = 1.0;
        ptrdiff_t ind = 0;
        int d;
        for(d = 0; d < pm->ndim; d++) {
            int r = rel[d];

            kernel *= k[d][r];

            int targetpos = ipos[d] + r;

            while(targetpos >= pm->Nmesh[d]) {
                targetpos -= pm->Nmesh[d];
            }
            while(targetpos < 0) {
                targetpos += pm->Nmesh[d];
            }
            if(UNLIKELY(targetpos >= pm->size[d])) {
                goto outside;
            }
            ind += strides[d] * targetpos;
        }
        value += kernel * canvas[ind];
outside:
        rel[pm->ndim - 1] ++;
        for(d = pm->ndim - 1; d > 0; d --) {
            if(UNLIKELY(rel[d] == s2)) {
                rel[d - 1] ++;
                rel[d] = 0;
            }
        }
    }
    return value;
}

