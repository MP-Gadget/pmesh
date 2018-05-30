from runtests.mpi import MPITest
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

def perturb_mass(mass, ind, value, comm):
    mass = mass.copy()
    start = sum(comm.allgather(mass.shape[0])[:comm.rank])
    end = sum(comm.allgather(mass.shape[0])[:comm.rank + 1])
    if ind >= start and ind < end:
        old = mass[ind - start]
        coord = mass[ind-start].copy()
        mass[ind - start] = old + value
        new = mass[ind - start]
    else:
        old = 0
        new = 0
        coord = 0
    diff = comm.allreduce(new - old)

    return diff, mass, comm.allreduce(coord)

def get_mass(mass, ind, comm):
    start = sum(comm.allgather(mass.shape[0])[:comm.rank])
    end = sum(comm.allgather(mass.shape[0])[:comm.rank + 1])
    if ind >= start and ind < end:
        old = mass[ind - start]
    else:
        old = 0
    return comm.allreduce(old)

@MPITest([1, 4])
def test_c2r_vjp(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    real = pm.generate_whitenoise(1234, type='real', mean=1.0)
    comp = real.r2c()

    def objective(comp):
        real = comp.c2r()
        obj = (real.value ** 2).sum()
        return comm.allreduce(obj)

    grad_real = RealField(pm)
    grad_real[...] = real[...] * 2
    grad_comp = ComplexField(pm)
    grad_comp = grad_real.c2r_vjp(grad_real)
    grad_comp.decompress_vjp(grad_comp)

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
def test_readout_gradients(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8', resampler='cic')

    real = pm.generate_whitenoise(1234, type='real')

    def objective(real, pos, layout):
        value = real.readout(pos, layout=layout)
        obj = (value ** 2).sum()
        return comm.allreduce(obj)

    def forward_gradient(real, pos, layout, v_real=None, v_pos=None):
        value = real.readout(pos, layout=layout)
        v_value = real.readout_jvp(pos, v_self=v_real, v_pos=v_pos, layout=layout)
        return comm.allreduce((v_value * value * 2).sum())

    def backward_gradient(real, pos, layout):
        value = real.readout(pos, layout=layout)
        return real.readout_vjp(pos, v=value * 2, layout=layout)

    pos = numpy.array(numpy.indices(real.shape), dtype='f8').reshape(real.value.ndim, -1).T
    pos += real.start
    # avoid sitting at the pmesh points
    # cic gradient is zero on them, the numerical gradient fails.
    pos += 0.5
    pos *= pm.BoxSize / pm.Nmesh

    layout = pm.decompose(pos)

    obj = objective(real, pos, layout)

    grad_real, grad_pos = backward_gradient(real, pos, layout)

    ng = []
    fag = []
    bag = []
    ind = []
    dx = 1e-6
    for ind1 in numpy.ndindex(*grad_real.cshape):
        dx1, r1 = perturb(real, ind1, dx)
        ng1 = (objective(r1, pos, layout) - obj)

        bag1 = grad_real.cgetitem(ind1) * dx1
        fag1 = forward_gradient(real, pos, layout, v_real=r1 - real)

        # print (dx1, dx, ind1, ag1, ng1)
        ng.append(ng1)
        fag.append(fag1)
        bag.append(bag1)
        ind.append(ind1)

    assert_allclose(bag, fag, rtol=1e-7)
    assert_allclose(ng, bag, rtol=1e-4)

    # FIXME
    ng = []
    bag = []
    fag = []
    ind = []
    dx = 1e-6

    for ind1 in numpy.ndindex((real.csize, real.ndim)):
        dx1, pos1, old = perturb_pos(pos, ind1, dx, comm)
        layout1 = pm.decompose(pos1)

        ng1 = (objective(real, pos1, layout1) - obj)
        bag1 = get_pos(grad_pos, ind1, comm) * dx1

        fag1 = forward_gradient(real, pos, layout, v_pos=pos1 - pos)

        # print ('pos', old, ind1, 'a', ag1, 'n', ng1)
        ng.append(ng1)
        bag.append(bag1)
        fag.append(fag1)
        ind.append(ind1)

    assert_allclose(bag, fag, rtol=1e-7)
    assert_allclose(ng, bag, rtol=1e-4)

@MPITest([1, 2])
def test_paint_gradients(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8', resampler='cic')

    real = pm.generate_whitenoise(1234, type='real')

    def objective(pos, mass, layout):
        real = pm.paint(pos, mass=mass, layout=layout)
        obj = (real[...] ** 2).sum()
        return comm.allreduce(obj)

    def forward_gradient(pos, mass, layout, v_pos=None, v_mass=None):
        real = pm.paint(pos, mass=mass, layout=layout)
        jvp = pm.paint_jvp(pos, mass=mass, v_mass=v_mass, v_pos=v_pos, layout=layout)
        return comm.allreduce((jvp * real * 2)[...].sum())

    def backward_gradient(pos, mass, layout):
        real = pm.paint(pos, mass=mass, layout=layout)
        return pm.paint_vjp(real * 2, pos, mass=mass, layout=layout)

    pos = numpy.array(numpy.indices(real.shape), dtype='f8').reshape(real.value.ndim, -1).T
    pos += real.start
    numpy.random.seed(9999)
    # avoid sitting at the pmesh points
    # cic gradient is zero on them, the numerical gradient fails.
    pos += (numpy.random.uniform(size=pos.shape)) * 0.8 + 0.1
    pos *= pm.BoxSize / pm.Nmesh
    mass = numpy.ones(len(pos)) * 2

    layout = pm.decompose(pos)

    obj = objective(pos, mass, layout)

    grad_pos, grad_mass = backward_gradient(pos, mass, layout)

    ng = []
    fag = []
    bag = []
    ind = []
    dx = 1e-6
    for ind1 in numpy.ndindex(real.csize):
        dx1, mass1, old = perturb_mass(mass, ind1[0], dx, comm)
        ng1 = (objective(pos, mass1, layout) - obj)

        bag1 = get_mass(grad_mass, ind1[0], comm) * dx1
        fag1 = forward_gradient(pos, mass, layout, v_mass=mass1 - mass)

    #    print (dx1, dx, ind1, fag1, bag1, ng1)
        ng.append(ng1)
        fag.append(fag1)
        bag.append(bag1)
        ind.append(ind1)

    assert_allclose(bag, fag, rtol=1e-7)
    assert_allclose(ng, bag, rtol=1e-4)

    # FIXME
    ng = []
    bag = []
    fag = []
    ind = []
    dx = 1e-6

    for ind1 in numpy.ndindex((real.csize, real.ndim)):
        dx1, pos1, old = perturb_pos(pos, ind1, dx, comm)
        layout1 = pm.decompose(pos1)

        ng1 = (objective(pos1, mass, layout1) - obj)
        bag1 = get_pos(grad_pos, ind1, comm) * dx1

        fag1 = forward_gradient(pos, mass, layout, v_pos=pos1 - pos)

    #    print ('pos', old, ind1, 'v_pos', (pos1 - pos)[ind1[0]], 'fag', fag1, 'bag', bag1, 'n', ng1)
        ng.append(ng1)
        bag.append(bag1)
        fag.append(fag1)
        ind.append(ind1)

    assert_allclose(bag, fag, rtol=1e-7)
    assert_allclose(ng, bag, rtol=1e-4)

@MPITest([1, 4])
def test_cdot_grad(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    comp1 = pm.generate_whitenoise(1234, type='complex', mean=1)
    comp2 = pm.generate_whitenoise(1235, type='complex', mean=1)

    def objective(comp1, comp2):
        return comp1.cdot(comp2).real

    y = objective(comp1, comp2)

    grad_comp2 = comp1.cdot_vjp(1.0)
    grad_comp1 = comp2.cdot_vjp(1.0)

    grad_comp1.decompress_vjp(grad_comp1)
    grad_comp2.decompress_vjp(grad_comp2)

    print(grad_comp1)
    print("comp1")
    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex(*(list(comp1.cshape) + [2])):
        dx1, c1 = perturb(comp1, ind1, dx)
        ng1 = (objective(c1, comp2) - objective(comp1, comp2)) / dx
        ag1 = grad_comp1.cgetitem(ind1) * dx1 / dx
        if abs(ag1 - ng1) > 1e-5 * max((abs(ag1), abs(ng1))):
            print (ind1, 'a', ag1, 'n', ng1)
        comm.barrier()
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5)

    print("comp2")

    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex(*(list(comp1.cshape) + [2])):
        dx1, c2 = perturb(comp2, ind1, dx)
        ng1 = (objective(comp1, c2) - objective(comp1, comp2)) / dx
        ag1 = grad_comp2.cgetitem(ind1) * dx1 / dx
        if abs(ag1 - ng1) > 1e-5 * max((abs(ag1), abs(ng1))):
            print (ind1, 'a', ag1, 'n', ng1)
        comm.barrier()
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5)

@MPITest([1, 4])
def test_cnorm_grad(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')

    comp1 = pm.generate_whitenoise(1234, type='complex')

    def objective(comp1):
        return comp1.cnorm()

    grad_comp1 = comp1 * 2
    grad_comp1.decompress_vjp(grad_comp1)

    ng = []
    ag = []
    ind = []
    dx = 1e-7
    for ind1 in numpy.ndindex(*(list(comp1.cshape) + [2])):
        dx1, c1 = perturb(comp1, ind1, dx)
        ng1 = (objective(c1) - objective(comp1)) / dx
        ag1 = grad_comp1.cgetitem(ind1) * dx1 / dx
        if abs(ag1 - ng1) > 1e-5 * max((abs(ag1), abs(ng1))):
            print (ind1, 'a', ag1, 'n', ng1)
        comm.barrier()
        ng.append(ng1)
        ag.append(ag1)
        ind.append(ind1)

    assert_allclose(ng, ag, rtol=1e-5, atol=1e-6)

