#define UNLIKELY(x) (x)
#define LIKELY(x) (x)

#include <stddef.h>
#include <stdint.h>
#include <math.h>

static inline void
mkname (_WRtPlus) (FLOAT * canvas, 
        const int i, const int j, const int k, const double f, const PMeshPainter * const painter)
{
    if(UNLIKELY(0 > i || painter->size[0] <= i)) return;
    if(UNLIKELY(0 > j || painter->size[1] <= j)) return;
    if(UNLIKELY(0 > k || painter->size[2] <= k)) return;
    ptrdiff_t ind = k * painter->strides[2] + j * painter->strides[1] + i * painter->strides[0];
#ifdef _OPENMP
#pragma omp atomic
#endif
    * (FLOAT*) ((char*) canvas + ind) += f;
    return;
}

static inline double 
mkname (_REd) (FLOAT const * const canvas, const int i, const int j, const int k, const double w, const PMeshPainter * const painter)
{
    if(UNLIKELY(0 > i || painter->size[0] <= i)) return 0;
    if(UNLIKELY(0 > j || painter->size[1] <= j)) return 0;
    if(UNLIKELY(0 > k || painter->size[2] <= k)) return 0;
    ptrdiff_t ind = k * painter->strides[2] + j * painter->strides[1] + i * painter->strides[0];
    return (* (FLOAT*) ((char*) canvas + ind)) * w;
}

#define ACCESS(func, a, b, c) \
    mkname(func)(canvas, IJK ## a [0], IJK ## b [1], IJK ## c [2], V ## a [0] * V ## b [1] * V ## c [2], painter)

#define SETUP_KERNEL_CIC \
    int d; \
    double XYZ[3]; \
    int IJK0[3], IJK1[3]; \
    double V1[3], V0[3]; \
\
    for(d = 0; d < 3; d ++) { \
        XYZ[d] = pos[d]* painter->scale[d] + painter->translate[d]; \
        /* without floor, -1 < X < 0 is mapped to I=0 */ \
        IJK0[d] = (int) floor(XYZ[d]); \
        IJK1[d] = IJK0[d] + 1; \
    }; \
\
    for(d = 0; d < 3; d ++) { \
        if(painter->order[d] == 0) { \
            V1[d] = XYZ[d] - IJK0[d]; \
            V0[d] = 1. - V1[d]; \
        } else { \
            V1[d] = painter->scale[d]; \
            V0[d] = - painter->scale[d]; \
        } \
    } \
\
    /* Do periodic wrapup in all directions. */ \
    /*  Buffer particles are copied from adjacent nodes */ \
    for(d = 0; d < 3; d ++) { \
        if(painter->Nmesh[d] == 0) continue; \
        while(UNLIKELY(IJK0[d] < 0)) IJK0[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK0[d] >= painter->Nmesh[d])) IJK0[d] -= painter->Nmesh[d]; \
        while(UNLIKELY(IJK1[d] < 0)) IJK1[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK1[d] >= painter->Nmesh[d])) IJK1[d] -= painter->Nmesh[d]; \
    } \

static void
mkname(_cic_tuned_paint) (PMeshPainter * painter, double pos[], double weight)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_CIC;

    V1[1] *= weight;
    V0[1] *= weight;

    ACCESS(_WRtPlus, 0, 0, 0);
    ACCESS(_WRtPlus, 0, 0, 1);
    ACCESS(_WRtPlus, 0, 1, 0);
    ACCESS(_WRtPlus, 0, 1, 1);
    ACCESS(_WRtPlus, 1, 0, 0);
    ACCESS(_WRtPlus, 1, 0, 1);
    ACCESS(_WRtPlus, 1, 1, 0);
    ACCESS(_WRtPlus, 1, 1, 1);
}

static double
mkname(_cic_tuned_readout) (PMeshPainter * painter, double pos[])
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_CIC;

    double value = 0;

    value += ACCESS(_REd, 0, 0, 0);
    value += ACCESS(_REd, 0, 0, 1);
    value += ACCESS(_REd, 0, 1, 0);
    value += ACCESS(_REd, 0, 1, 1);
    value += ACCESS(_REd, 1, 0, 0);
    value += ACCESS(_REd, 1, 0, 1);
    value += ACCESS(_REd, 1, 1, 0);
    value += ACCESS(_REd, 1, 1, 1);
    return value;
}
#undef SETUP_KERNEL_CIC

#define SETUP_KERNEL_TSC \
    int d; \
    double XYZ[3]; \
    int IJK0[3], IJK1[3], IJK2[3]; \
    double V0[3], V1[3], V2[3]; \
\
    for(d = 0; d < 3; d ++) { \
        XYZ[d] = pos[d]* painter->scale[d] + painter->translate[d]; \
        /* without floor, -1 < X < 0 is mapped to I=0 */ \
        IJK1[d] = (int) floor(XYZ[d] + 0.5); \
        IJK0[d] = IJK1[d] - 1; \
        IJK2[d] = IJK1[d] + 1; \
    }; \
\
    for(d = 0; d < 3; d ++) { \
        if(painter->order[d] == 0) { \
            V1[d] = 0.75 - (XYZ[d] - IJK1[d]) * (XYZ[d] - IJK1[d]); \
            V0[d] = (1.5 - (XYZ[d] - IJK0[d])) * (1.5 - (XYZ[d] - IJK0[d])) * 0.5; \
            V2[d] = (1.5 + (XYZ[d] - IJK2[d])) * (1.5 + (XYZ[d] - IJK2[d])) * 0.5; \
        } else { \
            V1[d] = -2 * (XYZ[d] - IJK1[d]) * painter->scale[d]; \
            V0[d] = - (1.5 - (XYZ[d] - IJK0[d])) * painter->scale[d]; \
            V2[d] = (1.5 + (XYZ[d] - IJK2[d])) * painter->scale[d]; \
        } \
    } \
\
    /* Do periodic wrapup in all directions.  */ \
    /* Buffer particles are copied from adjacent nodes */ \
    for(d = 0; d < 3; d ++) { \
        if(painter->Nmesh[d] == 0) continue; \
        while(UNLIKELY(IJK0[d] < 0)) IJK0[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK0[d] >= painter->Nmesh[d])) IJK0[d] -= painter->Nmesh[d]; \
        while(UNLIKELY(IJK1[d] < 0)) IJK1[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK1[d] >= painter->Nmesh[d])) IJK1[d] -= painter->Nmesh[d]; \
        while(UNLIKELY(IJK2[d] < 0)) IJK2[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK2[d] >= painter->Nmesh[d])) IJK2[d] -= painter->Nmesh[d]; \
    } \

static void
mkname(_tsc_tuned_paint) (PMeshPainter * painter, double pos[], double weight)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_TSC;

    V0[1] *= weight;
    V1[1] *= weight;
    V2[1] *= weight;

    ACCESS(_WRtPlus, 0, 0, 0);
    ACCESS(_WRtPlus, 0, 0, 1);
    ACCESS(_WRtPlus, 0, 0, 2);
    ACCESS(_WRtPlus, 0, 1, 0);
    ACCESS(_WRtPlus, 0, 1, 1);
    ACCESS(_WRtPlus, 0, 1, 2);
    ACCESS(_WRtPlus, 0, 2, 0);
    ACCESS(_WRtPlus, 0, 2, 1);
    ACCESS(_WRtPlus, 0, 2, 2);
    ACCESS(_WRtPlus, 1, 0, 0);
    ACCESS(_WRtPlus, 1, 0, 1);
    ACCESS(_WRtPlus, 1, 0, 2);
    ACCESS(_WRtPlus, 1, 1, 0);
    ACCESS(_WRtPlus, 1, 1, 1);
    ACCESS(_WRtPlus, 1, 1, 2);
    ACCESS(_WRtPlus, 1, 2, 0);
    ACCESS(_WRtPlus, 1, 2, 1);
    ACCESS(_WRtPlus, 1, 2, 2);
    ACCESS(_WRtPlus, 2, 0, 0);
    ACCESS(_WRtPlus, 2, 0, 1);
    ACCESS(_WRtPlus, 2, 0, 2);
    ACCESS(_WRtPlus, 2, 1, 0);
    ACCESS(_WRtPlus, 2, 1, 1);
    ACCESS(_WRtPlus, 2, 1, 2);
    ACCESS(_WRtPlus, 2, 2, 0);
    ACCESS(_WRtPlus, 2, 2, 1);
    ACCESS(_WRtPlus, 2, 2, 2);
}

static double
mkname(_tsc_tuned_readout) (PMeshPainter * painter, double pos[])
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_TSC;

    double value = 0;
    value += ACCESS(_REd, 0, 0, 0);
    value += ACCESS(_REd, 0, 0, 1);
    value += ACCESS(_REd, 0, 0, 2);
    value += ACCESS(_REd, 0, 1, 0);
    value += ACCESS(_REd, 0, 1, 1);
    value += ACCESS(_REd, 0, 1, 2);
    value += ACCESS(_REd, 0, 2, 0);
    value += ACCESS(_REd, 0, 2, 1);
    value += ACCESS(_REd, 0, 2, 2);
    value += ACCESS(_REd, 1, 0, 0);
    value += ACCESS(_REd, 1, 0, 1);
    value += ACCESS(_REd, 1, 0, 2);
    value += ACCESS(_REd, 1, 1, 0);
    value += ACCESS(_REd, 1, 1, 1);
    value += ACCESS(_REd, 1, 1, 2);
    value += ACCESS(_REd, 1, 2, 0);
    value += ACCESS(_REd, 1, 2, 1);
    value += ACCESS(_REd, 1, 2, 2);
    value += ACCESS(_REd, 2, 0, 0);
    value += ACCESS(_REd, 2, 0, 1);
    value += ACCESS(_REd, 2, 0, 2);
    value += ACCESS(_REd, 2, 1, 0);
    value += ACCESS(_REd, 2, 1, 1);
    value += ACCESS(_REd, 2, 1, 2);
    value += ACCESS(_REd, 2, 2, 0);
    value += ACCESS(_REd, 2, 2, 1);
    value += ACCESS(_REd, 2, 2, 2);
    return value;
}
#undef SETUP_KERNEL_TSC
#undef ACCESS
