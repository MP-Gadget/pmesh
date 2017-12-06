#define UNLIKELY(x) (x)
#define LIKELY(x) (x)

static void
mkname(_generic_paint) (PMeshPainter * painter, double pos[], double weight, double hsml)
{
    PMeshWindowInfo window[1];
    pmesh_window_info_init(window, painter->ndim, painter->nativesupport, painter->support * hsml);

    /* Check for fast painting routines */
    paintfunc fastpaint;
    readoutfunc fastreadout;

    if(painter->getfastmethod &&
       painter->getfastmethod(painter, window, &fastpaint, &fastreadout)) {

        fastpaint(painter, pos, weight, hsml);
        return;
    }

    int ipos[painter->ndim];

    /* the max support is 32 */
    double k[painter->ndim * window->support];

    char * canvas = (char*) painter->canvas;

    _fill_k(painter, window, pos, ipos, k);

    int rel[painter->ndim];
    int d;
    for(d =0; d < painter->ndim; d ++ ) rel[d] = 0;

    int s2 = window->support;
    while(rel[0] != s2) {
        double kernel = 1.0;
        ptrdiff_t ind = 0;
        int d;
        for(d = 0; d < painter->ndim; d++) {
            double * kd = &k[window->support * d];
            int r = rel[d];
            int targetpos = ipos[d] + r;
            kernel *= kd[r];
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
#ifdef _OPENMP
#pragma omp atomic
#endif
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

static double
mkname(_generic_readout) (PMeshPainter * painter, double pos[], double hsml)
{
    PMeshWindowInfo window[1];
    pmesh_window_info_init(window, painter->ndim, painter->nativesupport, painter->support * hsml);

    /* Check for fast painting routines */
    paintfunc fastpaint;
    readoutfunc fastreadout;

    if(painter->getfastmethod &&
       painter->getfastmethod(painter, window, &fastpaint, &fastreadout)) {

        return fastreadout(painter, pos, hsml);
    }

    double value = 0;
    int ipos[painter->ndim];
    double k[painter->ndim * window->support];

    char * canvas = (char*) painter->canvas;

    _fill_k(painter, window, pos, ipos, k);

    int rel[painter->ndim];
    int d;
    for(d =0; d < painter->ndim; d++) rel[d] = 0;

    int s2 = window->support;
    while(rel[0] != s2) {
        double kernel = 1.0;
        ptrdiff_t ind = 0;
        int d;
        for(d = 0; d < painter->ndim; d++) {
            double * kd = &k[window->support * d];
            int r = rel[d];

            kernel *= kd[r];

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

static inline void
mkname (_WRtPlus3) (FLOAT * canvas, 
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
mkname (_REd3) (FLOAT const * const canvas, const int i, const int j, const int k, const double w, const PMeshPainter * const painter)
{
    if(UNLIKELY(0 > i || painter->size[0] <= i)) return 0;
    if(UNLIKELY(0 > j || painter->size[1] <= j)) return 0;
    if(UNLIKELY(0 > k || painter->size[2] <= k)) return 0;
    ptrdiff_t ind = k * painter->strides[2] + j * painter->strides[1] + i * painter->strides[0];
    return (* (FLOAT*) ((char*) canvas + ind)) * w;
}

static inline void
mkname (_WRtPlus2) (FLOAT * canvas, 
        const int i, const int j, const double f, const PMeshPainter * const painter)
{
    if(UNLIKELY(0 > i || painter->size[0] <= i)) return;
    if(UNLIKELY(0 > j || painter->size[1] <= j)) return;
    ptrdiff_t ind = j * painter->strides[1] + i * painter->strides[0];
#ifdef _OPENMP
#pragma omp atomic
#endif
    * (FLOAT*) ((char*) canvas + ind) += f;
    return;
}

static inline double 
mkname (_REd2) (FLOAT const * const canvas, const int i, const int j, const double w, const PMeshPainter * const painter)
{
    if(UNLIKELY(0 > i || painter->size[0] <= i)) return 0;
    if(UNLIKELY(0 > j || painter->size[1] <= j)) return 0;
    ptrdiff_t ind = j * painter->strides[1] + i * painter->strides[0];
    return (* (FLOAT*) ((char*) canvas + ind)) * w;
}

static inline void
mkname (_WRtPlus1) (FLOAT * canvas, 
        const int i, const double f, const PMeshPainter * const painter)
{
    if(UNLIKELY(0 > i || painter->size[0] <= i)) return;
    ptrdiff_t ind = i * painter->strides[0];
#ifdef _OPENMP
#pragma omp atomic
#endif
    * (FLOAT*) ((char*) canvas + ind) += f;
    return;
}

static inline double 
mkname (_REd1) (FLOAT const * const canvas, const int i, const double w, const PMeshPainter * const painter)
{
    if(UNLIKELY(0 > i || painter->size[0] <= i)) return 0;
    ptrdiff_t ind = i * painter->strides[0];
    return (* (FLOAT*) ((char*) canvas + ind)) * w;
}

#define ACCESS3(func, a, b, c) \
    mkname(func)(canvas, IJK ## a [0], IJK ## b [1], IJK ## c [2], V ## a [0] * V ## b [1] * V ## c [2], painter)

#define ACCESS2(func, a, b) \
    mkname(func)(canvas, IJK ## a [0], IJK ## b [1], V ## a [0] * V ## b [1], painter)

#define ACCESS1(func, a) \
    mkname(func)(canvas, IJK ## a [0], V ## a [0], painter)
