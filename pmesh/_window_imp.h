#ifdef __cplusplus
extern "C" {
#endif
typedef enum { FASTPM_PAINTER_LINEAR, FASTPM_PAINTER_LANCZOS} FastPMPainterType;

typedef struct FastPMPainter FastPMPainter;
typedef struct PMInterface PMInterface;

typedef double (*fastpm_kernelfunc)(double x, double invh);

struct PMInterface {
    int ndim;

    ptrdiff_t Nmesh[32];
    double BoxSize[32];

    /* Local Storage */
    ptrdiff_t start[32];
    ptrdiff_t size[32];
    ptrdiff_t strides[32];
};

struct FastPMPainter {
    PMInterface * pm;
    double InvCellSize[32];

    struct {
        void   (*paint)(FastPMPainter * painter, double * canvas, ptrdiff_t strides[], double pos[], double weight, int diffdir);
        double (*readout)(FastPMPainter * painter, double * canvas, ptrdiff_t strides[], double pos[], int diffdir);
    } DOUBLE;
    struct {
        void   (*paint)(FastPMPainter * painter, float * canvas, ptrdiff_t strides[], double pos[], double weight, int diffdir);
        double (*readout)(FastPMPainter * painter, float * canvas, ptrdiff_t strides[], double pos[], int diffdir);
    } SINGLE;

    fastpm_kernelfunc kernel;
    fastpm_kernelfunc diff;

    int diffdir;
    int support;
    double hsupport;
    double invh;
    int left; /* offset to start the kernel, (support - 1) / 2*/
    int Npoints; /* (support) ** ndim */
};

void fastpm_painter_init(FastPMPainter * painter, PMInterface * pm,
        FastPMPainterType type, int support);

void fastpm_painter_init_diff(FastPMPainter * painter, FastPMPainter * base, int diffdir);

#ifdef __cplusplus
}
#endif
