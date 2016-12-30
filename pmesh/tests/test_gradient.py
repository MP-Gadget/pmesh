from mpi4py_test import MPITest
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal

from pmesh.pm import ParticleMesh, RealField, ComplexField
from pmesh import window
import numpy

def perturb(comp, mode, value):
    comp = comp.copy()
    old = comp.cgetitem(mode)
    new = comp.csetitem(mode, value + old)
    return new - old, comp

def perturb_pos(pos, ind, value, comm):
    pos = pos.copy()
    start = sum(comm.allgather(pos.shape[0])[:comm.rank])
    end = sum(comm.allgather(pos.shape[0])[:comm.rank + 1])
    if ind[0] >= start and ind[0] < end:
        old = pos[ind[0] - start, ind[1]]
        coord = pos[ind[0]-start].copy()
        pos[ind[0] - start, ind[1]] = old + value
        new = pos[ind[0] - start, ind[1]]
    else:
        old = 0
        new = 0
        coord = 0
    diff = comm.allreduce(new - old)

    return diff, pos, comm.allreduce(coord)
def get_pos(pos, ind, comm):
    start = sum(comm.allgather(pos.shape[0])[:comm.rank])
    end = sum(comm.allgather(pos.shape[0])[:comm.rank + 1])
    if ind[0] >= start and ind[0] < end:
        old = pos[ind[0] - start, ind[1]]
    else:
        old = 0
    return comm.allreduce(old)

@MPITest([1, 4])
def test_c2r_gradient(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    real = RealField(pm)
    real.generate_whitenoise(1234)
    comp = real.r2c()

    def objective(comp):
        real = comp.c2r()
        obj = (real.value ** 2).sum()
        return comm.allreduce(obj)

    grad_real = RealField(pm)
    grad_real[...] = real[...] * 2
    grad_comp = ComplexField(pm)
    grad_comp = grad_real.c2r_gradient(grad_real)
    grad_comp.decompress_gradient(grad_comp)

    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex(*(list(grad_comp.cshape) + [2])):
        dx1, c1 = perturb(comp, ind1, dx)
        ng1 = (objective(c1) - objective(comp)) / dx
        ag1 = grad_comp.cgetitem(ind1) * dx1 / dx
        comm.barrier()
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5)

@MPITest([1, 2])
def test_readout_gradient(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8', method='cic')

    real = RealField(pm)
    real.generate_whitenoise(1234)

    def objective(real, pos, layout):
        value = real.readout(pos, layout=layout)
        obj = (value ** 2).sum()
        return comm.allreduce(obj)

    pos = numpy.array(numpy.indices(real.shape), dtype='f4').reshape(real.value.ndim, -1).T
    pos += real.start
    # avoid sitting at the pmesh points
    # cic gradient is zero on them, the numerical gradient fails.
    pos += 0.5
    pos *= pm.BoxSize / pm.Nmesh

    layout = pm.decompose(pos)

    obj = objective(real, pos, layout)
    value = real.readout(pos, layout=layout)
    grad_value = value * 2
    grad_real, grad_pos = real.readout_gradient(pos, btgrad=grad_value, layout=layout)

    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex(*grad_real.cshape):
        dx1, r1 = perturb(real, ind1, dx)
        ng1 = (objective(r1, pos, layout) - obj)/dx
        ag1 = grad_real.cgetitem(ind1) * dx1 / dx
        # print (dx1, dx, ind1, ag1, ng1)
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5)

    # FIXME
    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex((real.csize, real.ndim)):
        dx1, pos1, old = perturb_pos(pos, ind1, dx, comm)
        layout1 = pm.decompose(pos1)
        ng1 = (objective(real, pos1, layout1) - obj)/dx
        ag1 = get_pos(grad_pos, ind1, comm) * dx1 / dx
        # print ('pos', old, ind1, 'a', ag1, 'n', ng1)
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5)

