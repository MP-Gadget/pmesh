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

#define FILL_CIC \
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

    FILL_CIC;

    V1[1] *= weight;
    V0[1] *= weight;

    mkname(_WRtPlus)(canvas, IJK0[0], IJK0[1], IJK0[2], V0[0]*V0[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK0[1], IJK1[2], V0[0]*V0[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK1[1], IJK0[2], V0[0]*V1[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK1[1], IJK1[2], V0[0]*V1[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK0[2], V1[0]*V0[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK1[2], V1[0]*V0[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK1[1], IJK0[2], V1[0]*V1[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK1[1], IJK1[2], V1[0]*V1[1]*V1[2], painter);
}

static double
mkname(_cic_tuned_readout) (PMeshPainter * painter, double pos[])
{
    FLOAT * canvas = painter->canvas;

    FILL_CIC;

    double value = 0;

    value += mkname(_REd)(canvas, IJK0[0], IJK0[1], IJK0[2], V0[0]*V0[1]*V0[2], painter);
    value += mkname(_REd)(canvas, IJK0[0], IJK0[1], IJK1[2], V0[0]*V0[1]*V1[2], painter);
    value += mkname(_REd)(canvas, IJK0[0], IJK1[1], IJK0[2], V0[0]*V1[1]*V0[2], painter);
    value += mkname(_REd)(canvas, IJK0[0], IJK1[1], IJK1[2], V0[0]*V1[1]*V1[2], painter);
    value += mkname(_REd)(canvas, IJK1[0], IJK0[1], IJK0[2], V1[0]*V0[1]*V0[2], painter);
    value += mkname(_REd)(canvas, IJK1[0], IJK0[1], IJK1[2], V1[0]*V0[1]*V1[2], painter);
    value += mkname(_REd)(canvas, IJK1[0], IJK1[1], IJK0[2], V1[0]*V1[1]*V0[2], painter);
    value += mkname(_REd)(canvas, IJK1[0], IJK1[1], IJK1[2], V1[0]*V1[1]*V1[2], painter);
    return value;
}
#undef FILL_CIC

#define FILL_TSC \
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

    FILL_TSC;

    V0[1] *= weight;
    V1[1] *= weight;
    V2[1] *= weight;

    mkname(_WRtPlus)(canvas, IJK0[0], IJK0[1], IJK0[2], V0[0]*V0[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK0[1], IJK1[2], V0[0]*V0[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK0[1], IJK2[2], V0[0]*V0[1]*V2[2], painter);

    mkname(_WRtPlus)(canvas, IJK0[0], IJK1[1], IJK0[2], V0[0]*V1[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK1[1], IJK1[2], V0[0]*V1[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK1[1], IJK2[2], V0[0]*V1[1]*V2[2], painter);

    mkname(_WRtPlus)(canvas, IJK0[0], IJK2[1], IJK0[2], V0[0]*V2[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK2[1], IJK1[2], V0[0]*V2[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK0[0], IJK2[1], IJK2[2], V0[0]*V2[1]*V2[2], painter);

    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK0[2], V1[0]*V0[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK1[2], V1[0]*V0[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK2[2], V1[0]*V0[1]*V2[2], painter);

    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK0[2], V1[0]*V1[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK1[2], V1[0]*V1[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK2[2], V1[0]*V1[1]*V2[2], painter);

    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK0[2], V2[0]*V2[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK1[2], V2[0]*V2[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK1[0], IJK0[1], IJK2[2], V2[0]*V2[1]*V2[2], painter);

    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK0[2], V2[0]*V0[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK1[2], V2[0]*V0[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK2[2], V2[0]*V0[1]*V2[2], painter);

    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK0[2], V2[0]*V1[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK1[2], V2[0]*V1[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK2[2], V2[0]*V1[1]*V2[2], painter);

    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK0[2], V2[0]*V2[1]*V0[2], painter);
    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK1[2], V2[0]*V2[1]*V1[2], painter);
    mkname(_WRtPlus)(canvas, IJK2[0], IJK0[1], IJK2[2], V2[0]*V2[1]*V2[2], painter);
}

static double
mkname(_tsc_tuned_readout) (PMeshPainter * painter, double pos[])
{
    FLOAT * canvas = painter->canvas;

    FILL_TSC;

    double value = 0;
    value += mkname(_Red)(canvas, IJK0[0], IJK0[1], IJK0[2], V0[0]*V0[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK0[0], IJK0[1], IJK1[2], V0[0]*V0[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK0[0], IJK0[1], IJK2[2], V0[0]*V0[1]*V2[2], painter);

    value += mkname(_Red)(canvas, IJK0[0], IJK1[1], IJK0[2], V0[0]*V1[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK0[0], IJK1[1], IJK1[2], V0[0]*V1[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK0[0], IJK1[1], IJK2[2], V0[0]*V1[1]*V2[2], painter);

    value += mkname(_Red)(canvas, IJK0[0], IJK2[1], IJK0[2], V0[0]*V2[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK0[0], IJK2[1], IJK1[2], V0[0]*V2[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK0[0], IJK2[1], IJK2[2], V0[0]*V2[1]*V2[2], painter);

    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK0[2], V1[0]*V0[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK1[2], V1[0]*V0[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK2[2], V1[0]*V0[1]*V2[2], painter);

    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK0[2], V1[0]*V1[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK1[2], V1[0]*V1[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK2[2], V1[0]*V1[1]*V2[2], painter);

    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK0[2], V2[0]*V2[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK1[2], V2[0]*V2[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK1[0], IJK0[1], IJK2[2], V2[0]*V2[1]*V2[2], painter);

    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK0[2], V2[0]*V0[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK1[2], V2[0]*V0[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK2[2], V2[0]*V0[1]*V2[2], painter);

    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK0[2], V2[0]*V1[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK1[2], V2[0]*V1[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK2[2], V2[0]*V1[1]*V2[2], painter);

    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK0[2], V2[0]*V2[1]*V0[2], painter);
    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK1[2], V2[0]*V2[1]*V1[2], painter);
    value += mkname(_Red)(canvas, IJK2[0], IJK0[1], IJK2[2], V2[0]*V2[1]*V2[2], painter);
    return value;
}
#undef FILL_TSC
