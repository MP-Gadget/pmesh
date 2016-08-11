from mpi4py_test import MPIWorld
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal

from pmesh.pm import ParticleMesh, RealField, ComplexField

import numpy

@MPIWorld(NTask=(1, 4), required=(1))
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
    real[:] = 0
    real.paint(npos)
    real2 = real.copy()
    real.r2c(complex)

    real[...] = 0
    complex.c2r(real)
    real.readout(npos)
    assert_almost_equal(real, real2)

@MPIWorld(NTask=(1), required=(1))
def test_indices(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[2, 2], comm=comm, dtype='f8')
    assert_almost_equal(pm.k[0], [[0], [-0.785398]], decimal=3)
    assert_almost_equal(pm.k[1], [[0, -0.785398]], decimal=3)
    assert_almost_equal(pm.x[0], [[0], [-4]], decimal=3)
    assert_almost_equal(pm.x[1], [[0, -4]], decimal=3)

@MPIWorld(NTask=(1, 4), required=(1))
def test_real_iter(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)

    for x, slab in real.slabiter():
        assert slab.base is real
        assert_array_equal(slab.shape, sum(x[d] ** 2 for d in range(len(pm.Nmesh))).shape)

@MPIWorld(NTask=(1, 2, 3, 4), required=(1))
def test_sort(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 6], comm=comm, dtype='f8')
    real = RealField(pm)
    truth = numpy.arange(8 * 6)
    real[...] = truth.reshape(8, 6)[real.slices]
    real.sort()
    conjecture = numpy.concatenate(comm.allgather(real.ravel()))
    assert_array_equal(conjecture, truth)

    complex = ComplexField(pm)
    truth = numpy.arange(8 * 4)
    complex[...] = truth.reshape(8, 4)[complex.slices]
    complex.sort()
    conjecture = numpy.concatenate(comm.allgather(complex.ravel()))
    assert_array_equal(conjecture, truth)

@MPIWorld(NTask=(1, 2, 3, 4), required=(1))
def test_downsample(comm):
    pm1 = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    pm2 = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    complex1 = ComplexField(pm1)
    complex2 = ComplexField(pm2)
    for kk, slab in complex1.slabiter():
        slab[...] = sum([k**2 for k in kk]) **0.5

    complex1.resample(complex2)

    assert_array_equal(complex2, sum([k**2 for k in complex2.x]) **0.5)

@MPIWorld(NTask=(1, 2, 3, 4), required=(1))
def test_upsample(comm):
    pm1 = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8, 8], comm=comm, dtype='f8')
    pm2 = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')

    complex1 = ComplexField(pm1)
    complex2 = ComplexField(pm2)
    for kk, slab in complex1.slabiter():
        slab[...] = sum([k**2 for k in kk]) **0.5

    complex1.resample(complex2)
    complex2.resample(complex1)
    complex1.resample(complex2)

    assert_array_equal(complex2, sum([k**2 for k in complex2.x]) **0.5)

@MPIWorld(NTask=(1, 4), required=1)
def test_complex_iter(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    complex = ComplexField(pm)

    for x, slab in complex.slabiter():
        assert_array_equal(slab.shape, sum(x[d] ** 2 for d in range(len(pm.Nmesh))).shape)
