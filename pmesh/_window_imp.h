#ifdef __cplusplus
extern "C" {
#endif
typedef enum { PMESH_PAINTER_NEAREST,
               PMESH_PAINTER_LINEAR,
               PMESH_PAINTER_CUBIC,
               PMESH_PAINTER_QUADRATIC,
               PMESH_PAINTER_LANCZOS2,
               PMESH_PAINTER_LANCZOS3,
               PMESH_PAINTER_LANCZOS4,
               PMESH_PAINTER_LANCZOS5,
               PMESH_PAINTER_LANCZOS6,
               PMESH_PAINTER_ACG2, /* Approximated Confined Gaussian */
               PMESH_PAINTER_ACG3,
               PMESH_PAINTER_ACG4,
               PMESH_PAINTER_ACG5,
               PMESH_PAINTER_ACG6,
               PMESH_PAINTER_DB6,
               PMESH_PAINTER_DB12,
               PMESH_PAINTER_DB20,
               PMESH_PAINTER_SYM6,
               PMESH_PAINTER_SYM12,
               PMESH_PAINTER_SYM20,
               PMESH_PAINTER_TUNED_NNB,
               PMESH_PAINTER_TUNED_CIC,
               PMESH_PAINTER_TUNED_TSC,
               PMESH_PAINTER_TUNED_PCS,
} PMeshPainterType;

typedef struct PMeshWindowInfo {
    int support;
    double vfactor; /* nativesupport / support */
    double shift;
    int left; /* offset to start the kernel, (support - 1) / 2*/
    int Npoints; /* (support) ** ndim */
} PMeshWindowInfo;

typedef struct PMeshPainter PMeshPainter;

typedef double (*pmesh_kernelfunc)(double x);
typedef double (*pmesh_fwindowfunc)(double w);

typedef    void   (*paintfunc)(PMeshPainter * painter, double pos[], double weight, double hsml);
typedef    double (*readoutfunc)(PMeshPainter * painter, double pos[], double hsml);

typedef int (*getfastmethodfunc)(PMeshPainter * painter, PMeshWindowInfo * window, paintfunc * paint, readoutfunc * readout);

struct PMeshPainter {
    PMeshPainterType type;
    int order[32]; /* diff order per axis */
    int ndim;
    double scale[32]; /* scale from position to grid units */
    double translate[32]; /* translate in grid units */
    ptrdiff_t Nmesh[32]; /* periodicity */
    int support;

    void * canvas;
    int canvas_dtype_elsize;
    ptrdiff_t size[32];
    ptrdiff_t strides[32];

    /* Private: */
    paintfunc paint;
    readoutfunc readout;
    getfastmethodfunc getfastmethod;

    pmesh_kernelfunc kernel;
    pmesh_kernelfunc diff;
    pmesh_fwindowfunc fwindow; /* fourier space window */

    double nativesupport; /* unscaled support */

    PMeshWindowInfo window;
};

void
pmesh_painter_init(PMeshPainter * painter);

void
pmesh_painter_paint(PMeshPainter * painter, double pos[], double weight, double hsml);

double
pmesh_painter_readout(PMeshPainter * painter, double pos[], double hsml);

double
pmesh_painter_get_fwindow(PMeshPainter * painter, double w);

#ifdef __cplusplus
}
#endif
