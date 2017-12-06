#define SETUP_KERNEL_PCS(Nd) \
    int d; \
    double XYZ[Nd]; \
    int IJK0[Nd], IJK1[Nd], IJK2[Nd], IJK3[Nd]; \
    double V0[Nd], V1[Nd], V2[Nd], V3[Nd]; \
\
    for(d = 0; d < Nd; d ++) { \
        XYZ[d] = pos[d]* painter->scale[d] + painter->translate[d]; \
        /* without floor, -1 < X < 0 is mapped to I=0 */ \
        IJK1[d] = (int) floor(XYZ[d]); \
        IJK0[d] = IJK1[d] - 1; \
        IJK2[d] = IJK1[d] + 1; \
        IJK3[d] = IJK2[d] + 1; \
    }; \
\
    for(d = 0; d < Nd; d ++) { \
        if(painter->order[d] == 0) { \
            V1[d] = 1.0 / 6.0 * (4 - \
                    6 * (XYZ[d] - IJK1[d]) * (XYZ[d] - IJK1[d]) \
                  + 3 * (XYZ[d] - IJK1[d]) *(XYZ[d] - IJK1[d]) * (XYZ[d] - IJK1[d])); \
            V2[d] = 1.0 / 6.0 * (4 - \
                    6 * (XYZ[d] - IJK2[d]) * (XYZ[d] - IJK2[d]) \
                  - 3 * (XYZ[d] - IJK2[d]) *(XYZ[d] - IJK2[d]) * (XYZ[d] - IJK2[d])); \
            V0[d] = 1.0 / 6.0 * (2 - (XYZ[d] - IJK0[d])) \
                              * (2 - (XYZ[d] - IJK0[d])) \
                              * (2 - (XYZ[d] - IJK0[d])); \
            V3[d] = 1.0 / 6.0 * (2 + (XYZ[d] - IJK3[d])) \
                              * (2 + (XYZ[d] - IJK3[d])) \
                              * (2 + (XYZ[d] - IJK3[d])); \
        } else { \
            V1[d] = + 1.0 / 6.0 * (- 12 * (XYZ[d] - IJK1[d]) \
                                   + 9 * (XYZ[d] - IJK1[d]) * (XYZ[d] - IJK1[d])); \
            V2[d] = - 1.0 / 6.0 * (+ 12 * (XYZ[d] - IJK2[d]) \
                                   + 9 * (XYZ[d] - IJK2[d]) * (XYZ[d] - IJK2[d])); \
            V0[d] = - 1.0 / 2.0 * (2 - (XYZ[d] - IJK0[d])) * (2 - (XYZ[d] - IJK0[d])); \
            V3[d] = + 1.0 / 2.0 * (2 + (XYZ[d] - IJK3[d])) * (2 + (XYZ[d] - IJK3[d])); \
        } \
    } \
\
    /* Do periodic wrapup in all directions. */ \
    /*  Buffer particles are copied from adjacent nodes */ \
    for(d = 0; d < Nd; d ++) { \
        if(painter->Nmesh[d] == 0) continue; \
        while(UNLIKELY(IJK0[d] < 0)) IJK0[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK0[d] >= painter->Nmesh[d])) IJK0[d] -= painter->Nmesh[d]; \
        while(UNLIKELY(IJK1[d] < 0)) IJK1[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK1[d] >= painter->Nmesh[d])) IJK1[d] -= painter->Nmesh[d]; \
        while(UNLIKELY(IJK2[d] < 0)) IJK2[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK2[d] >= painter->Nmesh[d])) IJK2[d] -= painter->Nmesh[d]; \
        while(UNLIKELY(IJK3[d] < 0)) IJK3[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK3[d] >= painter->Nmesh[d])) IJK3[d] -= painter->Nmesh[d]; \
    } \

static void
mkname(_pcs_tuned_paint3) (PMeshPainter * painter, double pos[], double weight, double hsml)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_PCS(3);

    V0[0] *= weight;
    V1[0] *= weight;
    V2[0] *= weight;
    V3[0] *= weight;

    ACCESS3(_WRtPlus3, 0, 0, 0);
    ACCESS3(_WRtPlus3, 0, 0, 1);
    ACCESS3(_WRtPlus3, 0, 0, 2);
    ACCESS3(_WRtPlus3, 0, 0, 3);
    ACCESS3(_WRtPlus3, 0, 1, 0);
    ACCESS3(_WRtPlus3, 0, 1, 1);
    ACCESS3(_WRtPlus3, 0, 1, 2);
    ACCESS3(_WRtPlus3, 0, 1, 3);
    ACCESS3(_WRtPlus3, 0, 2, 0);
    ACCESS3(_WRtPlus3, 0, 2, 1);
    ACCESS3(_WRtPlus3, 0, 2, 2);
    ACCESS3(_WRtPlus3, 0, 2, 3);
    ACCESS3(_WRtPlus3, 0, 3, 0);
    ACCESS3(_WRtPlus3, 0, 3, 1);
    ACCESS3(_WRtPlus3, 0, 3, 2);
    ACCESS3(_WRtPlus3, 0, 3, 3);
    ACCESS3(_WRtPlus3, 1, 0, 0);
    ACCESS3(_WRtPlus3, 1, 0, 1);
    ACCESS3(_WRtPlus3, 1, 0, 2);
    ACCESS3(_WRtPlus3, 1, 0, 3);
    ACCESS3(_WRtPlus3, 1, 1, 0);
    ACCESS3(_WRtPlus3, 1, 1, 1);
    ACCESS3(_WRtPlus3, 1, 1, 2);
    ACCESS3(_WRtPlus3, 1, 1, 3);
    ACCESS3(_WRtPlus3, 1, 2, 0);
    ACCESS3(_WRtPlus3, 1, 2, 1);
    ACCESS3(_WRtPlus3, 1, 2, 2);
    ACCESS3(_WRtPlus3, 1, 2, 3);
    ACCESS3(_WRtPlus3, 1, 3, 0);
    ACCESS3(_WRtPlus3, 1, 3, 1);
    ACCESS3(_WRtPlus3, 1, 3, 2);
    ACCESS3(_WRtPlus3, 1, 3, 3);
    ACCESS3(_WRtPlus3, 2, 0, 0);
    ACCESS3(_WRtPlus3, 2, 0, 1);
    ACCESS3(_WRtPlus3, 2, 0, 2);
    ACCESS3(_WRtPlus3, 2, 0, 3);
    ACCESS3(_WRtPlus3, 2, 1, 0);
    ACCESS3(_WRtPlus3, 2, 1, 1);
    ACCESS3(_WRtPlus3, 2, 1, 2);
    ACCESS3(_WRtPlus3, 2, 1, 3);
    ACCESS3(_WRtPlus3, 2, 2, 0);
    ACCESS3(_WRtPlus3, 2, 2, 1);
    ACCESS3(_WRtPlus3, 2, 2, 2);
    ACCESS3(_WRtPlus3, 2, 2, 3);
    ACCESS3(_WRtPlus3, 2, 3, 0);
    ACCESS3(_WRtPlus3, 2, 3, 1);
    ACCESS3(_WRtPlus3, 2, 3, 2);
    ACCESS3(_WRtPlus3, 2, 3, 3);
    ACCESS3(_WRtPlus3, 3, 0, 0);
    ACCESS3(_WRtPlus3, 3, 0, 1);
    ACCESS3(_WRtPlus3, 3, 0, 2);
    ACCESS3(_WRtPlus3, 3, 0, 3);
    ACCESS3(_WRtPlus3, 3, 1, 0);
    ACCESS3(_WRtPlus3, 3, 1, 1);
    ACCESS3(_WRtPlus3, 3, 1, 2);
    ACCESS3(_WRtPlus3, 3, 1, 3);
    ACCESS3(_WRtPlus3, 3, 2, 0);
    ACCESS3(_WRtPlus3, 3, 2, 1);
    ACCESS3(_WRtPlus3, 3, 2, 2);
    ACCESS3(_WRtPlus3, 3, 2, 3);
    ACCESS3(_WRtPlus3, 3, 3, 0);
    ACCESS3(_WRtPlus3, 3, 3, 1);
    ACCESS3(_WRtPlus3, 3, 3, 2);
    ACCESS3(_WRtPlus3, 3, 3, 3);
}

static double
mkname(_pcs_tuned_readout3) (PMeshPainter * painter, double pos[], double hsml)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_PCS(3);

    double value = 0;

    value += ACCESS3(_REd3, 0, 0, 0);
    value += ACCESS3(_REd3, 0, 0, 1);
    value += ACCESS3(_REd3, 0, 0, 2);
    value += ACCESS3(_REd3, 0, 0, 3);
    value += ACCESS3(_REd3, 0, 1, 0);
    value += ACCESS3(_REd3, 0, 1, 1);
    value += ACCESS3(_REd3, 0, 1, 2);
    value += ACCESS3(_REd3, 0, 1, 3);
    value += ACCESS3(_REd3, 0, 2, 0);
    value += ACCESS3(_REd3, 0, 2, 1);
    value += ACCESS3(_REd3, 0, 2, 2);
    value += ACCESS3(_REd3, 0, 2, 3);
    value += ACCESS3(_REd3, 0, 3, 0);
    value += ACCESS3(_REd3, 0, 3, 1);
    value += ACCESS3(_REd3, 0, 3, 2);
    value += ACCESS3(_REd3, 0, 3, 3);
    value += ACCESS3(_REd3, 1, 0, 0);
    value += ACCESS3(_REd3, 1, 0, 1);
    value += ACCESS3(_REd3, 1, 0, 2);
    value += ACCESS3(_REd3, 1, 0, 3);
    value += ACCESS3(_REd3, 1, 1, 0);
    value += ACCESS3(_REd3, 1, 1, 1);
    value += ACCESS3(_REd3, 1, 1, 2);
    value += ACCESS3(_REd3, 1, 1, 3);
    value += ACCESS3(_REd3, 1, 2, 0);
    value += ACCESS3(_REd3, 1, 2, 1);
    value += ACCESS3(_REd3, 1, 2, 2);
    value += ACCESS3(_REd3, 1, 2, 3);
    value += ACCESS3(_REd3, 1, 3, 0);
    value += ACCESS3(_REd3, 1, 3, 1);
    value += ACCESS3(_REd3, 1, 3, 2);
    value += ACCESS3(_REd3, 1, 3, 3);
    value += ACCESS3(_REd3, 2, 0, 0);
    value += ACCESS3(_REd3, 2, 0, 1);
    value += ACCESS3(_REd3, 2, 0, 2);
    value += ACCESS3(_REd3, 2, 0, 3);
    value += ACCESS3(_REd3, 2, 1, 0);
    value += ACCESS3(_REd3, 2, 1, 1);
    value += ACCESS3(_REd3, 2, 1, 2);
    value += ACCESS3(_REd3, 2, 1, 3);
    value += ACCESS3(_REd3, 2, 2, 0);
    value += ACCESS3(_REd3, 2, 2, 1);
    value += ACCESS3(_REd3, 2, 2, 2);
    value += ACCESS3(_REd3, 2, 2, 3);
    value += ACCESS3(_REd3, 2, 3, 0);
    value += ACCESS3(_REd3, 2, 3, 1);
    value += ACCESS3(_REd3, 2, 3, 2);
    value += ACCESS3(_REd3, 2, 3, 3);
    value += ACCESS3(_REd3, 3, 0, 0);
    value += ACCESS3(_REd3, 3, 0, 1);
    value += ACCESS3(_REd3, 3, 0, 2);
    value += ACCESS3(_REd3, 3, 0, 3);
    value += ACCESS3(_REd3, 3, 1, 0);
    value += ACCESS3(_REd3, 3, 1, 1);
    value += ACCESS3(_REd3, 3, 1, 2);
    value += ACCESS3(_REd3, 3, 1, 3);
    value += ACCESS3(_REd3, 3, 2, 0);
    value += ACCESS3(_REd3, 3, 2, 1);
    value += ACCESS3(_REd3, 3, 2, 2);
    value += ACCESS3(_REd3, 3, 2, 3);
    value += ACCESS3(_REd3, 3, 3, 0);
    value += ACCESS3(_REd3, 3, 3, 1);
    value += ACCESS3(_REd3, 3, 3, 2);
    value += ACCESS3(_REd3, 3, 3, 3);
    return value;
}

static void
mkname(_pcs_tuned_paint2) (PMeshPainter * painter, double pos[], double weight, double hsml)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_PCS(2);

    V0[0] *= weight;
    V1[0] *= weight;
    V2[0] *= weight;
    V3[0] *= weight;

    ACCESS2(_WRtPlus2, 0, 0);
    ACCESS2(_WRtPlus2, 0, 1);
    ACCESS2(_WRtPlus2, 0, 2);
    ACCESS2(_WRtPlus2, 0, 3);
    ACCESS2(_WRtPlus2, 1, 0);
    ACCESS2(_WRtPlus2, 1, 1);
    ACCESS2(_WRtPlus2, 1, 2);
    ACCESS2(_WRtPlus2, 1, 3);
    ACCESS2(_WRtPlus2, 2, 0);
    ACCESS2(_WRtPlus2, 2, 1);
    ACCESS2(_WRtPlus2, 2, 2);
    ACCESS2(_WRtPlus2, 2, 3);
    ACCESS2(_WRtPlus2, 3, 0);
    ACCESS2(_WRtPlus2, 3, 1);
    ACCESS2(_WRtPlus2, 3, 2);
    ACCESS2(_WRtPlus2, 3, 3);
}

static double
mkname(_pcs_tuned_readout2) (PMeshPainter * painter, double pos[], double hsml)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_PCS(2);

    double value = 0;

    value += ACCESS2(_REd2, 0, 0);
    value += ACCESS2(_REd2, 0, 1);
    value += ACCESS2(_REd2, 0, 2);
    value += ACCESS2(_REd2, 0, 3);
    value += ACCESS2(_REd2, 1, 0);
    value += ACCESS2(_REd2, 1, 1);
    value += ACCESS2(_REd2, 1, 2);
    value += ACCESS2(_REd2, 1, 3);
    value += ACCESS2(_REd2, 2, 0);
    value += ACCESS2(_REd2, 2, 1);
    value += ACCESS2(_REd2, 2, 2);
    value += ACCESS2(_REd2, 2, 3);
    value += ACCESS2(_REd2, 3, 0);
    value += ACCESS2(_REd2, 3, 1);
    value += ACCESS2(_REd2, 3, 2);
    value += ACCESS2(_REd2, 3, 3);
    return value;
}

static void
mkname(_pcs_tuned_paint1) (PMeshPainter * painter, double pos[], double weight, double hsml)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_PCS(1);

    V0[0] *= weight;
    V1[0] *= weight;
    V2[0] *= weight;
    V3[0] *= weight;

    ACCESS1(_WRtPlus1, 0);
    ACCESS1(_WRtPlus1, 1);
    ACCESS1(_WRtPlus1, 2);
    ACCESS1(_WRtPlus1, 3);
}

static double
mkname(_pcs_tuned_readout1) (PMeshPainter * painter, double pos[], double hsml)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_PCS(1);

    double value = 0;

    value += ACCESS1(_REd1, 0);
    value += ACCESS1(_REd1, 1);
    value += ACCESS1(_REd1, 2);
    value += ACCESS1(_REd1, 3);
    return value;
}
static int
mkname(_getfastmethod_pcs) (PMeshPainter * painter, PMeshWindowInfo * window, paintfunc * fastpaint, readoutfunc * fastreadout)
{
    if(window->support != 4) return 0;

    if(painter->ndim == 1) {
        *fastpaint = mkname(_pcs_tuned_paint1); \
        *fastreadout = mkname(_pcs_tuned_readout1); \
        return 1;
    }
    if(painter->ndim == 2) {
        *fastpaint = mkname(_pcs_tuned_paint2); \
        *fastreadout = mkname(_pcs_tuned_readout2); \
        return 1;
    }
    if(painter->ndim == 3) {
        *fastpaint = mkname(_pcs_tuned_paint3); \
        *fastreadout = mkname(_pcs_tuned_readout3); \
        return 1;
    }
    return 0;
}

#undef SETUP_KERNEL_PCS

