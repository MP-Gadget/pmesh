#define UNLIKELY(x) (x)
#define LIKELY(x) (x)

#include <stddef.h>
#include <stdint.h>
#include <math.h>

static inline void
mkname (_WRtPlus) (FLOAT * canvas, 
        const int i, const int j, const int k, const double f, ptrdiff_t strides[3])
{
    ptrdiff_t ind = k * strides[2] + j * strides[1] + i * strides[0];
#pragma omp atomic
    canvas[ind] += f;
    return;
}

static inline double 
mkname (_REd) (FLOAT const * const canvas, const int i, const int j, const int k, const double w, ptrdiff_t strides[3])
{
    ptrdiff_t ind = k * strides[2] + j * strides[1] + i * strides[0];
    return canvas[ind] * w;
}

static void
mkname(_cic_normal_paint) (PMeshPainter * painter, double pos[], double weight)
{
    int d;

    double XYZ[3];
    int IJK[3];
    int IJK1[3];
    double D[3];
    double T[3];

    FLOAT * canvas = painter->canvas;

    for(d = 0; d < 3; d ++) {
        XYZ[d] = pos[d]* painter->scale[d] + painter->translate[d];
        // without floor, -1 < X < 0 is mapped to I=0
        IJK[d] = (int) floor(XYZ[d]);
        IJK1[d] = IJK[d] + 1;
    };

    for(d = 0; d < 3; d ++) {
        D[d] = XYZ[d] - IJK[d];
        T[d] = 1. - D[d];
    }

    // Do periodic wrapup in all directions. 
    // Buffer particles are copied from adjacent nodes
    for(d = 0; d < 3; d ++) {
        while(UNLIKELY(IJK[d] < 0)) IJK[d] += painter->Nmesh[d];
        while(UNLIKELY(IJK[d] >= painter->Nmesh[d])) IJK[d] -= painter->Nmesh[d];
        while(UNLIKELY(IJK1[d] < 0)) IJK1[d] += painter->Nmesh[d];
        while(UNLIKELY(IJK1[d] >= painter->Nmesh[d])) IJK1[d] -= painter->Nmesh[d];
    }

    D[1] *= weight;
    T[1] *= weight;

    if(LIKELY(0 <= IJK[0] && IJK[0] < painter->size[0])) {
        if(LIKELY(0 <= IJK[1] && IJK[1] < painter->size[1])) {
            if(LIKELY(0 <= IJK[2] && IJK[2] < painter->size[2]))
                mkname(_WRtPlus)(canvas, IJK[0], IJK[1],  IJK[2],  T[2]*T[0]*T[1], painter);
            if(LIKELY(0 <= IJK1[2] && IJK1[2] < painter->size[2]))
                mkname(_WRtPlus)(canvas, IJK[0], IJK[1],  IJK1[2], D[2]*T[0]*T[1], painter);
        }
        if(LIKELY(0 <= IJK1[1] && IJK1[1] < painter->size[1])) {
            if(LIKELY(0 <= IJK[2] && IJK[2] < painter->size[2]))
                mkname(_WRtPlus)(canvas, IJK[0], IJK1[1], IJK[2],  T[2]*T[0]*D[1], painter);
            if(LIKELY(0 <= IJK1[2] && IJK1[2] < painter->size[2]))
                mkname(_WRtPlus)(canvas, IJK[0], IJK1[1], IJK1[2], D[2]*T[0]*D[1], painter);
        }
    }

    if(LIKELY(0 <= IJK1[0] && IJK1[0] < painter->size[0])) {
        if(LIKELY(0 <= IJK[1] && IJK[1] < painter->size[1])) {
            if(LIKELY(0 <= IJK[2] && IJK[2] < painter->size[2]))
                mkname(_WRtPlus)(canvas, IJK1[0], IJK[1],  IJK[2],  T[2]*D[0]*T[1], painter);
            if(LIKELY(0 <= IJK1[2] && IJK1[2] < painter->size[2]))
                mkname(_WRtPlus)(canvas, IJK1[0], IJK[1],  IJK1[2], D[2]*D[0]*T[1], painter);
        }
        if(LIKELY(0 <= IJK1[1] && IJK1[1] < painter->size[1])) {
            if(LIKELY(0 <= IJK[2] && IJK[2] < painter->size[2]))
                mkname(_WRtPlus)(canvas, IJK1[0], IJK1[1], IJK[2],  T[2]*D[0]*D[1], painter);
            if(LIKELY(0 <= IJK1[2] && IJK1[2] < painter->size[2]))
                mkname(_WRtPlus)(canvas, IJK1[0], IJK1[1], IJK1[2], D[2]*D[0]*D[1], painter);
        }
    }
}

static double
mkname(_generic_normal_readout) (PMeshPainter * painter, double pos[])
{
    int d;

    double XYZ[3];
    int IJK[3];
    int IJK1[3];
    double D[3];
    double T[3];

    FLOAT * canvas = painter->canvas;

    for(d = 0; d < 3; d ++) {
        XYZ[d] = pos[d]* painter->scale[d] + painter->translate[d];
        // without floor, -1 < X < 0 is mapped to I=0
        IJK[d] = (int) floor(XYZ[d]);
        IJK1[d] = IJK[d] + 1;
    };

    for(d = 0; d < 3; d ++) {
        D[d] = XYZ[d] - IJK[d];
        T[d] = 1. - D[d];
    }

    // Do periodic wrapup in all directions. 
    // Buffer particles are copied from adjacent nodes
    for(d = 0; d < 3; d ++) {
        while(UNLIKELY(IJK[d] < 0)) IJK[d] += painter->Nmesh[d];
        while(UNLIKELY(IJK[d] >= painter->Nmesh[d])) IJK[d] -= painter->Nmesh[d];
        while(UNLIKELY(IJK1[d] < 0)) IJK1[d] += painter->Nmesh[d];
        while(UNLIKELY(IJK1[d] >= painter->Nmesh[d])) IJK1[d] -= painter->Nmesh[d];
    }

    double value = 0;

    if(LIKELY(0 <= IJK[0] && IJK[0] < painter->size[0])) {
        if(LIKELY(0 <= IJK[1] && IJK[1] < painter->size[1])) {
            if(LIKELY(0 <= IJK[2] && IJK[2] < painter->size[2]))
                value += mkname(_REd)(canvas, IJK[0], IJK[1],  IJK[2],  T[2]*T[0]*T[1], painter);
            if(LIKELY(0 <= IJK1[2] && IJK1[2] < painter->size[2]))
                value += mkname(_REd)(canvas, IJK[0], IJK[1],  IJK1[2], D[2]*T[0]*T[1], painter);
        }
        if(LIKELY(0 <= IJK1[1] && IJK1[1] < painter->size[1])) {
            if(LIKELY(0 <= IJK[2] && IJK[2] < painter->size[2]))
                value += mkname(_REd)(canvas, IJK[0], IJK1[1], IJK[2],  T[2]*T[0]*D[1], painter);
            if(LIKELY(0 <= IJK1[2] && IJK1[2] < painter->size[2]))
                value += mkname(_REd)(canvas, IJK[0], IJK1[1], IJK1[2], D[2]*T[0]*D[1], painter);
        }
    }

    if(LIKELY(0 <= IJK1[0] && IJK1[0] < painter->size[0])) {
        if(LIKELY(0 <= IJK[1] && IJK[1] < painter->size[1])) {
            if(LIKELY(0 <= IJK[2] && IJK[2] < painter->size[2]))
                value += mkname(_REd)(canvas, IJK1[0], IJK[1],  IJK[2],  T[2]*D[0]*T[1], painter);
            if(LIKELY(0 <= IJK1[2] && IJK1[2] < painter->size[2]))
                value += mkname(_REd)(canvas, IJK1[0], IJK[1],  IJK1[2], D[2]*D[0]*T[1], painter);
        }
        if(LIKELY(0 <= IJK1[1] && IJK1[1] < painter->size[1])) {
            if(LIKELY(0 <= IJK[2] && IJK[2] < painter->size[2]))
                value += mkname(_REd)(canvas, IJK1[0], IJK1[1], IJK[2],  T[2]*D[0]*D[1], painter);
            if(LIKELY(0 <= IJK1[2] && IJK1[2] < painter->size[2]))
                value += mkname(_REd)(canvas, IJK1[0], IJK1[1], IJK1[2], D[2]*D[0]*D[1], painter);
        }
    }
    return value;
}
