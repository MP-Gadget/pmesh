from mpi4py_test import MPIWorld
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal

from pmesh.pm import ParticleMesh, RealField, ComplexField
import numpy

@MPIWorld(NTask=(1, 4), required=(1, 4))
def test_fft(comm):
    pm = ParticleMesh(BoxSize=10.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)
    complex = ComplexField(pm)
    numpy.random.seed(1234)
    if comm.rank == 0:
        Npar = 0
    else:
        Npar = 10

    pos = numpy.random.uniform(size=(Npar, len(pm.Nmesh))) * pm.BoxSize

    real.paint(pos)
    real2 = real.copy()

    real.r2c(complex)
    real[...] = 0
    complex.c2r(real)

    assert_almost_equal(real, real2)
