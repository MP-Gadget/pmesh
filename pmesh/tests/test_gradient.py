from mpi4py_test import MPIWorld
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal

from pmesh.pm import ParticleMesh, RealField, ComplexField
from pmesh import window
import numpy

def perturb(comp, mode, value):
    old = comp.plain[mode]
    comp.plain[mode] += value
    ret = comp.c2r().r2c()
    comp.plain[mode] = old
    return (ret.plain[mode] - comp.plain[mode]), ret

def rperturb(real, mode, value):
    old = real[mode]
    real[mode] += value
    ret = real.copy()
    real[mode] = old
    return (ret[mode] - real[mode]), ret

@MPIWorld(NTask=(1,), required=(1))
def test_c2r(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    real = RealField(pm)

    numpy.random.seed(1234)

    real[...] = numpy.random.normal(size=real.shape)
    comp = real.r2c()
    def objective(comp):
        real = comp.c2r()
        obj = (real.value ** 2).sum()
        return obj
    grad_real = RealField(pm)
    grad_real[...] = real[...] * 2

    grad_comp = ComplexField(pm)
    grad_comp = grad_real.c2r_gradient(grad_real)
    grad_comp.decompress_gradient(grad_comp)

    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex(*grad_comp.plain.shape):
        dx1, c1 = perturb(comp, ind1, dx)
        ng1 = (objective(c1) - objective(comp))/dx
        ag1 = grad_comp.plain[ind1] * dx1 / dx
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5)

@MPIWorld(NTask=(1,), required=(1))
def test_readout(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    real = RealField(pm)

    numpy.random.seed(1234)

    real[...] = numpy.random.normal(size=real.shape)

    def objective(real, pos):
        value = real.readout(pos)
        obj = (value ** 2).sum()
        return obj

    pos = numpy.array(numpy.indices(real.shape), dtype='f8').reshape(real.value.ndim, -1).T
    pos += 1.1
    pos *= pm.BoxSize / pm.Nmesh

    obj = objective(real, pos)

    value = real.readout(pos)
    grad_value = value * 2
    grad_real, grad_pos = real.readout_gradient(pos, btgrad=grad_value)

    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex(*grad_real.shape):
        dx1, r1 = rperturb(real, ind1, dx)
        ng1 = (objective(r1, pos) - objective(real, pos))/dx
        ag1 = grad_real[ind1] * dx1 / dx
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5)

    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex(*grad_pos.shape):
        dx1, pos1 = rperturb(pos, ind1, dx)
        ng1 = (objective(real, pos1) - objective(real, pos))/dx
        ag1 = grad_pos[ind1] * dx1 / dx
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5)

