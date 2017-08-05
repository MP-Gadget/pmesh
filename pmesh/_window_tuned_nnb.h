#define SETUP_KERNEL_NNB(Nd) \
    int d; \
    double XYZ[Nd]; \
    int IJK0[Nd]; \
    double V0[Nd]; \
\
    for(d = 0; d < Nd; d ++) { \
        XYZ[d] = pos[d] * painter->scale[d] + painter->translate[d]; \
        /* without floor, -1 < X < 0 is mapped to I=0 */ \
        IJK0[d] = (int) floor(XYZ[d] + 0.5); \
    }; \
\
    for(d = 0; d < Nd; d ++) { \
        if(painter->order[d] == 0) { \
            V0[d] = 1; \
        } else { \
            V0[d] = 0; \
        } \
    } \
\
    /* Do periodic wrapup in all directions. */ \
    /*  Buffer particles are copied from adjacent nodes */ \
    for(d = 0; d < Nd; d ++) { \
        if(painter->Nmesh[d] == 0) continue; \
        while(UNLIKELY(IJK0[d] < 0)) IJK0[d] += painter->Nmesh[d]; \
        while(UNLIKELY(IJK0[d] >= painter->Nmesh[d])) IJK0[d] -= painter->Nmesh[d]; \
    } \

static void
mkname(_nnb_tuned_paint3) (PMeshPainter * painter, double pos[], double weight)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_NNB(3);

    V0[1] *= weight;

    ACCESS3(_WRtPlus3, 0, 0, 0);
}

static double
mkname(_nnb_tuned_readout3) (PMeshPainter * painter, double pos[])
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_NNB(3);

    double value = 0;

    value += ACCESS3(_REd3, 0, 0, 0);
    return value;
}

static void
mkname(_nnb_tuned_paint2) (PMeshPainter * painter, double pos[], double weight)
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_NNB(2);
    V0[1] *= weight;

    ACCESS2(_WRtPlus2, 0, 0);
}

static double
mkname(_nnb_tuned_readout2) (PMeshPainter * painter, double pos[])
{
    FLOAT * canvas = painter->canvas;

    SETUP_KERNEL_NNB(2);

    double value = 0;

    value += ACCESS2(_REd2, 0, 0);
    return value;
}

#undef SETUP_KERNEL_NNB

