#define UNLIKELY(x) (x)
#define LIKELY(x) (x)

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include "_window_imp.h"

#include "_window_wavelets.h"
#include "_window_lanczos.h"
#include "_window_acg.h"

static double
sinc_unnormed(double x) {
    /* sin(x) / x */
    if(x < 1e-5 && x > -1e-5) {
        double x2 = x * x;
        return 1.0 - x2 / 6. + x2  * x2 / 120.;
    } else {
        return sin(x) / x;
    }
}

static void
pmesh_window_info_init(PMeshWindowInfo * info, int ndim, int nativesupport, double support)
{

    if(support <= 0) {
        info->support = nativesupport;
        support = nativesupport;
    } else {
        info->support = support;
        /* floating point rounding up */
        info->support += (support != (double)info->support);
    }
    info->left = (info->support - 1) / 2;

    info->shift = support / 2.0 - info->support / 2;

    int nmax = 1;
    int d;
    for(d = 0; d < ndim; d++) {
        nmax *= (info->support);
    }
    info->Npoints = nmax;
    info->vfactor = nativesupport / (1. * support);
}


static void
_fill_k(PMeshPainter * painter, PMeshWindowInfo * window, double pos[], int ipos[], double * k)
{
    double gpos[painter->ndim];
    int d;
    for(d = 0; d < painter->ndim; d++) {
        double * kd = &k[window->support * d];

        gpos[d] = pos[d] * painter->scale[d] + painter->translate[d];
        ipos[d] = floor(gpos[d] + window->shift) - window->left;
        double dx = gpos[d] - ipos[d]; /* relative to the left most nonzero.*/
        int i;
        for(i = 0; i < window->support; i ++) {
            double x = (dx - i) * window->vfactor;
            if(painter->order[d] == 0) {
                kd[i] = painter->kernel(x) * window->vfactor;
            } else {
                kd[i] = painter->diff(x) * painter->scale[d] * window->vfactor * window->vfactor;
            }
            //printf("i = %d x = %g kd[i] = %g\n", i, x, kd[i]);
        }
        /* Watch out: do not renormalize per particle */

        /* We require the kernels to be properly normalized instead,
         * because the sampling here is too coarse for individual renormalizing to make sense.
         * -- the normalization is very different for different offsets.
         */

        /*
         * the total mass of individual point is not supposed to conserve when we resample an
         * image. Nevertheless when we add them all up the total is statistically conserved.
         */
    }
}

#define FLOAT float
#define mkname(a) a ## _ ## float
#include "_window_generics.h"
#include "_window_tuned_nnb.h"
#include "_window_tuned_cic.h"
#include "_window_tuned_tsc.h"
#include "_window_tuned_pcs.h"
#undef FLOAT
#undef mkname
#undef ACCESS3
#undef ACCESS2
#define mkname(a) a ## _ ## double
#define FLOAT double
#include "_window_generics.h"
#include "_window_tuned_nnb.h"
#include "_window_tuned_cic.h"
#include "_window_tuned_tsc.h"
#include "_window_tuned_pcs.h"
#undef FLOAT
#undef mkname
#undef ACCESS3
#undef ACCESS2

static double
_nearest_kernel(double x) {
    /* implementiong nearest in floating is poor due to round-off errors */
    if(x < 0.5 && x >= - 0.5)
        return 1.0;
    return 0;
}

static double
_nearest_diff(double x) {
    return 0;
}

static double
_nearest_fwindow(double w) {
    /* p = 1 ; https://arxiv.org/abs/astro-ph/0409240 eq 18. */
    double t = sinc_unnormed(0.5 * w);
    return t;
}

static double
_linear_kernel(double x) {
    /* CIC */
    x = fabs(x);
    if(x < 1.0)
        return 1.0 - x;
    return 0;
}

static double
_linear_diff(double x) {
    double factor;
    if(x < 0) {
        factor = 1;
        x = - x;
    } else if (x > 0) {
        factor = - 1;
    } else {
        factor = 0;
    }
    if(x < 1.0) return factor;
    return 0;
}

static double
_linear_fwindow(double w) {
    /* p == 2*/
    double t = sinc_unnormed(0.5 * w);
    t = t * t;
    return t;
}

static double
_quadratic_kernel(double x) {
    /*
     * Take from https://arxiv.org/abs/0804.0070 TSC
     * */
    x = fabs(x);
    if(x <= 0.5) {
        return 0.75 - x * x;
    }
    if(x < 1.5) {
        x = 1.5 - x;
        return (x * x) * 0.5;
    }
    return 0;
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

    if(x <= 0.5) {
        return factor * (- 2 * x);
    }
    if(x < 1.5) {
        return factor * (- (1.5 - x));
    }
    return 0;
}

static double
_quadratic_fwindow(double w) {
    /* p = 3 ; https://arxiv.org/abs/astro-ph/0409240 eq 18. */
    double t = sinc_unnormed(0.5 * w);
    t = t * t * t;
    return t;
}


static double
_cubic_kernel(double x) {
    /* Take from eq 18 https://arxiv.org/pdf/1512.07295.pdf, PCS */
    x = fabs(x);
    double xx = x * x;
    if(x < 1.0) {
        return 1.0 / 6.0 * (4 - 6 * xx + 3 * xx * x);
    }
    if (x < 2) {
        return 1.0 / 6.0 * (2 - x) * (2 - x) * (2 - x);
    }
    return 0;
}

static double
_cubic_diff(double x) {
    double factor;
    if (x < 0) {
        factor = -1;
        x = -x;
    } else {
        factor = +1;
    }

    double xx = x * x;
    if(x < 1.0) {
        return factor * (1.0 / 6.0) * (- 12 * x + 9 * xx);
    }
    if(x < 2.0) {
        return factor * (- 1.0 / 2.0) * (2 - x) * (2 - x);
    }
    return 0;
}

static double
_cubic_fwindow(double w) {
    /* p = 4 ; https://arxiv.org/abs/astro-ph/0409240 eq 18. */
    double t = sinc_unnormed(0.5 * w);
    t = t * t * t * t;
    return t;
}

void
pmesh_painter_init(PMeshPainter * painter)
{
    painter->getfastmethod = NULL;

    if(painter->canvas_dtype_elsize == 8) {
        painter->paint = _generic_paint_double;
        painter->readout = _generic_readout_double;
    } else {
        painter->paint = _generic_paint_float;
        painter->readout = _generic_readout_float;
    }

    switch(painter->type) {
        case PMESH_PAINTER_NEAREST:
            painter->kernel = _nearest_kernel;
            painter->diff = _nearest_diff;
            painter->fwindow = _nearest_fwindow;
            painter->nativesupport = 1;
        break;
        case PMESH_PAINTER_LINEAR:
            painter->kernel = _linear_kernel;
            painter->diff = _linear_diff;
            painter->fwindow = _linear_fwindow;
            painter->nativesupport = 2;
        break;
        case PMESH_PAINTER_QUADRATIC:
            painter->kernel = _quadratic_kernel;
            painter->diff = _quadratic_diff;
            painter->fwindow = _quadratic_fwindow;
            painter->nativesupport = 3;
        break;
        case PMESH_PAINTER_CUBIC:
            painter->kernel = _cubic_kernel;
            painter->diff = _cubic_diff;
            painter->fwindow = _cubic_fwindow;
            painter->nativesupport = 4;
        break;
        case PMESH_PAINTER_LANCZOS2:
            painter->kernel = _lanczos2_kernel;
            painter->diff = _lanczos2_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _lanczos2_nativesupport;
        break;
        case PMESH_PAINTER_LANCZOS3:
            painter->kernel = _lanczos3_kernel;
            painter->diff = _lanczos3_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _lanczos3_nativesupport;
        break;
        case PMESH_PAINTER_LANCZOS4:
            painter->kernel = _lanczos4_kernel;
            painter->diff = _lanczos4_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _lanczos4_nativesupport;
        break;
        case PMESH_PAINTER_LANCZOS5:
            painter->kernel = _lanczos5_kernel;
            painter->diff = _lanczos5_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _lanczos5_nativesupport;
        break;
        case PMESH_PAINTER_LANCZOS6:
            painter->kernel = _lanczos6_kernel;
            painter->diff = _lanczos6_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _lanczos6_nativesupport;
        break;
        case PMESH_PAINTER_ACG2:
            painter->kernel = _acg2_kernel;
            painter->diff = _acg2_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _acg2_nativesupport;
        break;
        case PMESH_PAINTER_ACG3:
            painter->kernel = _acg3_kernel;
            painter->diff = _acg3_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _acg3_nativesupport;
        break;
        case PMESH_PAINTER_ACG4:
            painter->kernel = _acg4_kernel;
            painter->diff = _acg4_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _acg4_nativesupport;
        break;
        case PMESH_PAINTER_ACG5:
            painter->kernel = _acg5_kernel;
            painter->diff = _acg5_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _acg5_nativesupport;
        break;
        case PMESH_PAINTER_ACG6:
            painter->kernel = _acg6_kernel;
            painter->diff = _acg6_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _acg6_nativesupport;
        break;
        case PMESH_PAINTER_DB6:
            painter->kernel = _db6_kernel;
            painter->diff = _db6_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _db6_nativesupport;
        break;
        case PMESH_PAINTER_DB12:
            painter->kernel = _db12_kernel;
            painter->diff = _db12_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _db12_nativesupport;
        break;
        case PMESH_PAINTER_DB20:
            painter->kernel = _db20_kernel;
            painter->diff = _db20_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _db20_nativesupport;
        break;
        case PMESH_PAINTER_SYM6:
            painter->kernel = _sym6_kernel;
            painter->diff = _sym6_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _sym6_nativesupport;
        break;
        case PMESH_PAINTER_SYM12:
            painter->kernel = _sym12_kernel;
            painter->diff = _sym12_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _sym12_nativesupport;
        break;
        case PMESH_PAINTER_SYM20:
            painter->kernel = _sym20_kernel;
            painter->diff = _sym20_diff;
            painter->fwindow = NULL;
            painter->nativesupport = _sym20_nativesupport;
        break;
        case PMESH_PAINTER_TUNED_NNB:
            /* fall back to use nearest kernel */
            painter->kernel = _nearest_kernel;
            painter->diff = _nearest_diff;
            painter->fwindow = _nearest_fwindow;
            painter->nativesupport = 1;

            if(painter->order[0] > 1) break;
            if(painter->ndim > 1 && painter->order[1] > 1) break;
            if(painter->ndim > 2 && painter->order[2] > 1) break;
            if(painter->ndim > 3) break;

            if(painter->canvas_dtype_elsize == 8) {
                painter->getfastmethod = _getfastmethod_nnb_double;
            } else {
                painter->getfastmethod = _getfastmethod_nnb_float;
            }
        break;
        case PMESH_PAINTER_TUNED_CIC:
            /* fall back to use linear kernel */
            painter->kernel = _linear_kernel;
            painter->diff = _linear_diff;
            painter->fwindow = _linear_fwindow;
            painter->nativesupport = 2;

            if(painter->order[0] > 1) break;
            if(painter->ndim > 1 && painter->order[1] > 1) break;
            if(painter->ndim > 2 && painter->order[2] > 1) break;
            if(painter->ndim > 3) break;

            if(painter->canvas_dtype_elsize == 8) {
                painter->getfastmethod = _getfastmethod_cic_double;
            } else {
                painter->getfastmethod = _getfastmethod_cic_float;
            }
        break;
        case PMESH_PAINTER_TUNED_TSC:
            /* fall back to use quad kernel */
            painter->kernel = _quadratic_kernel;
            painter->diff = _quadratic_diff;
            painter->fwindow = _quadratic_fwindow;
            painter->nativesupport = 3;

            if(painter->order[0] > 1) break;
            if(painter->ndim > 1 && painter->order[1] > 1) break;
            if(painter->ndim > 2 && painter->order[2] > 1) break;
            if(painter->ndim > 3) break;

            if(painter->canvas_dtype_elsize == 8) {
                painter->getfastmethod = _getfastmethod_tsc_double;
            } else {
                painter->getfastmethod = _getfastmethod_tsc_float;
            }
        break;
        case PMESH_PAINTER_TUNED_PCS:
            /* fall back to use cubic kernel*/
            painter->kernel = _cubic_kernel;
            painter->diff = _cubic_diff;
            painter->fwindow = _cubic_fwindow;
            painter->nativesupport = 4;

            if(painter->order[0] > 1) break;
            if(painter->ndim > 1 && painter->order[1] > 1) break;
            if(painter->ndim > 2 && painter->order[2] > 1) break;
            if(painter->ndim > 3) break;

            /* nothing implemented yet */
            if(painter->canvas_dtype_elsize == 8) {
                painter->getfastmethod = _getfastmethod_pcs_double;
            } else {
                painter->getfastmethod = _getfastmethod_pcs_float;
            }
        break;
    }
    pmesh_window_info_init(&painter->window, painter->ndim, painter->nativesupport, painter->support);

    /* update the support */
    painter->support = painter->window.support;

}

void
pmesh_painter_paint(PMeshPainter * painter, double pos[], double weight, double hsml)
{
    painter->paint(painter, pos, weight, hsml);
}

double
pmesh_painter_readout(PMeshPainter * painter, double pos[], double hsml)
{
    return painter->readout(painter, pos, hsml);
}

double
pmesh_painter_get_fwindow(PMeshPainter * painter, double w)
{
    /* if support is larger, the window is shallower, thus fwindow shall be peakier.
     *  we shall look up from a larger frequency */
    if(painter->fwindow) {
        double r = painter->fwindow(w / painter->window.vfactor);
        return r;
    }
    else {
        return 1.0; /* not implemented */
    }
}

