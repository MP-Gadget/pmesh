#include <Python.h>

unsigned long int gsl_rng_default_seed = 0;

void
gsl_error (const char * reason, const char * file, int line, int gsl_errno)
{
    char str[1024];
    sprintf("%s : %d : errno = %d :%s", file, line, gsl_errno, reason);

    PyErr_SetString(PyExc_RuntimeError, str);
}
