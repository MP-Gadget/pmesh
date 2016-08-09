from mpi4py_test import MPIWorld
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal

from pmesh.pm import ParticleMesh, RealField, ComplexField

import numpy

@MPIWorld(NTask=(1, 4), required=(1, 4))
def test_fft(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)
    complex = ComplexField(pm)
    numpy.random.seed(1234)
    if comm.rank == 0:
        Npar = 10
    else:
        Npar = 0

    pos = 1.0 * (numpy.arange(Npar * len(pm.Nmesh))).reshape(-1, len(pm.Nmesh)) * (7, 7)
    pos %= (pm.Nmesh + 1)
    layout = pm.decompose(pos)

    npos = layout.exchange(pos)
    real.paint(npos)
    print real
    real2 = real.copy()
    real.r2c(complex)
    real[...] = 0
    complex.c2r(real)
    real.readout(npos)
    assert_almost_equal(real, real2)

@MPIWorld(NTask=(1, 4), required=(1, 4))
def test_real_iter(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)

    for x, slab in real.slabiter():
        assert slab.base is real
        slab[...] = sum(x[d] ** 2 for d in range(len(pm.Nmesh)))

@MPIWorld(NTask=(1, 4), required=(1, 4))
def test_complex_iter(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    complex = ComplexField(pm)

    for x, slab in complex.slabiter():
        assert slab.base is complex
        slab[...] = sum(x[d] ** 2 for d in range(len(pm.Nmesh)))
