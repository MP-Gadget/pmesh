from runtests.mpi import MPITest

import numpy
from pmesh import domain
from numpy.testing import assert_allclose, assert_array_equal

@MPITest(commsize=[1, 2, 3, 4])
def test_uniform(comm):
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND.uniform(
            BoxSize=[1, 2, 2],
            comm=comm,
            periodic=True)

    if comm.size == 4:
        assert_array_equal(dcop.shape, (1, 2, 2))

    if comm.size == 3:
        assert_array_equal(dcop.shape, (1, 3, 1))

    if comm.size == 2:
        assert_array_equal(dcop.shape, (1, 2, 1))

    if comm.size == 1:
        assert_array_equal(dcop.shape, (1, 1, 1))

@MPITest(commsize=4)
def test_extra_ranks(comm):
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    if comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((2, 2))), dtype='f8')
        mass = [0, 1, 2, 3]
    else:
        pos = numpy.empty((0, 2), dtype='f8')
        mass = []

    layout = dcop.decompose(pos, smoothing=0)
    assert_array_equal(layout.get_exchange_cost(),
        [2, 0, 0, 0])

    sendcounts = comm.allgather(layout.sendcounts)
    npos = layout.exchange(pos)
    npos = comm.allgather(npos)
    assert_array_equal(npos[0], [[0, 0], [0, 1]])
    assert_array_equal(npos[1], [[1, 0], [1, 1]])

    nmass = layout.exchange(mass)
    mass2 = layout.gather(nmass)
    nmass = comm.allgather(nmass)

    assert_array_equal(nmass[0], [0, 1])
    assert_array_equal(nmass[1], [2, 3])
    assert_array_equal(mass2, mass)

@MPITest(commsize=2)
def test_exchange(comm):
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    if comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((2, 2))), dtype='f8')
        mass = [0, 1, 2, 3]
    else:
        pos = numpy.empty((0, 2), dtype='f8')
        mass = []

    layout = dcop.decompose(pos, smoothing=0)
    sendcounts = comm.allgather(layout.sendcounts)
    npos = layout.exchange(pos)
    npos = comm.allgather(npos)
    assert_array_equal(npos[0], [[0, 0], [0, 1]])
    assert_array_equal(npos[1], [[1, 0], [1, 1]])

    nmass = layout.exchange(mass)
    mass2 = layout.gather(nmass)
    nmass = comm.allgather(nmass)

    assert_array_equal(nmass[0], [0, 1])
    assert_array_equal(nmass[1], [2, 3])
    assert_array_equal(mass2, mass)

@MPITest(commsize=2)
def test_exchange_struct(comm):
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    if comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((2, 2))), dtype='f8')
        mass = [0, 1, 2, 3]
    else:
        pos = numpy.empty((0, 2), dtype='f8')
        mass = []

    data = numpy.empty(len(pos), dtype=[('pos', ('f8', 2)), ('mass', 'f8')])
    data['pos'] = pos
    data['mass'] = mass
    layout = dcop.decompose(pos, smoothing=0)

    data = layout.exchange(data)
    npos = comm.allgather(data['pos'])
    assert_array_equal(npos[0], [[0, 0], [0, 1]])
    assert_array_equal(npos[1], [[1, 0], [1, 1]])

@MPITest(commsize=2)
def test_inhomotypes(comm):
    """ Testing type promotion """
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    if comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((2, 2))), dtype='f8')
        mass = numpy.array([0, 1, 2, 3], dtype='complex64')
    else:
        pos = numpy.empty((0, 2), dtype='f4')
        mass = numpy.array([], dtype='f8')

    layout = dcop.decompose(pos, smoothing=0)
    sendcounts = comm.allgather(layout.sendcounts)
    npos = layout.exchange(pos)
    assert npos.dtype == numpy.dtype('f8')
    npos = comm.allgather(npos)
    assert_array_equal(npos[0], [[0, 0], [0, 1]])
    assert_array_equal(npos[1], [[1, 0], [1, 1]])

    nmass = layout.exchange(mass)
    assert nmass.dtype == numpy.dtype('complex64')
    nmass = comm.allgather(nmass)
    assert_array_equal(nmass[0], [0, 1])
    assert_array_equal(nmass[1], [2, 3])

@MPITest(commsize=2)
def test_packed(comm):
    """ Testing type promotion of a packed exchange."""
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    if comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((2, 2))), dtype='f8')
        mass = numpy.array([0, 1, 2, 3], dtype='complex64')
    else:
        pos = numpy.empty((0, 2), dtype='f4')
        mass = numpy.array([], dtype='f8')

    layout = dcop.decompose(pos, smoothing=0)
    sendcounts = comm.allgather(layout.sendcounts)

    nposu, nmassu = layout.exchange(pos, mass, pack=False)
    assert nposu.dtype == numpy.dtype('f8')
    assert nmassu.dtype == numpy.dtype('complex64')

    npos, nmass = layout.exchange(pos, mass, pack=True)
    assert npos.dtype == numpy.dtype('f8')
    assert nmass.dtype == numpy.dtype('complex64')

    assert_array_equal(npos, nposu)
    assert_array_equal(nmass, nmassu)

    npos = comm.allgather(npos)
    nmass = comm.allgather(nmass)
    assert_array_equal(npos[0], [[0, 0], [0, 1]])
    assert_array_equal(npos[1], [[1, 0], [1, 1]])
    assert_array_equal(nmass[0], [0, 1])
    assert_array_equal(nmass[1], [2, 3])



@MPITest(commsize=3)
def test_period_empty_ranks(comm):
    DomainGrid = [[0, 2, 4, 4], [0, 4]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    pos = numpy.array([(0, 0)])
    layout = dcop.decompose(pos, smoothing=1.5)
    p1 = layout.exchange(pos)

    print(dcop.primary_region, len(p1))

    if comm.rank == 2:
        assert len(p1) == 0
    if comm.rank == 0:
        assert len(p1) == comm.size
    if comm.rank == 1:
        assert len(p1) == comm.size

@MPITest(commsize=4)
def test_period(comm):
    DomainGrid = [[0, 2, 4, 4], [0, 4]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    pos = numpy.array([(0, 0), (-1, -1)])
    layout = dcop.decompose(pos, smoothing=0.0)
    p1 = layout.exchange(pos)

    if comm.rank == 0:
        assert len(p1) == 4
    if comm.rank == 1:
        assert len(p1) == 4

@MPITest(commsize=2)
def test_exchange_smooth(comm):
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    if comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((2, 2))), dtype='f8')
    else:
        pos = numpy.empty((0, 2), dtype='f8')

    layout = dcop.decompose(pos, smoothing=1)
    sendcounts = comm.allgather(layout.sendcounts)
    npos = layout.exchange(pos)
    nmass = numpy.ones(len(npos))

    mass_sum = layout.gather(nmass, mode='sum')
    # because of the smoothing, every particle shall be repeated once
    assert_array_equal(mass_sum, 2)

    mass_any = layout.gather(nmass, mode='any')
    assert_array_equal(mass_any, 1)

    mass_fmax = layout.gather(nmass, mode=numpy.fmax)
    assert_array_equal(mass_fmax, 1)

    mass_fmin = layout.gather(nmass, mode=numpy.fmin)
    assert_array_equal(mass_fmin, 1)

    lpos = layout.gather(npos, mode='local')
    assert_array_equal(lpos, pos)

    npos = comm.allgather(npos)
    assert_array_equal(npos[0], [[0, 0], [0, 1], [1, 0], [1, 1]])
    assert_array_equal(npos[1], [[0, 0], [0, 1], [1, 0], [1, 1]])

@MPITest(commsize=2)
def test_isprimary(comm):
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    if comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((6, 6, 1))), dtype='f8')
        pos -= 2
    else:
        pos = numpy.empty((0, 3), dtype='f8')

    layout = dcop.decompose(pos, smoothing=1.5)
    npos = layout.exchange(pos)

    isprimary = dcop.isprimary(npos)
#    print('-----', comm.rank, isprimary, npos[isprimary], npos[~isprimary], dcop.primary_region)
    assert comm.allreduce(isprimary.sum()) == comm.allreduce(len(pos))

@MPITest(commsize=2)
def test_load(comm):
    DomainGrid = [[0, 1, 2], [0, 2]]

    dcop = domain.GridND(DomainGrid, 
            comm=comm,
            periodic=True)

    if comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((3, 6, 1))), dtype='f8')
        #pos -= 2
    else:
        pos = numpy.array(list(numpy.ndindex((6, 6, 1))), dtype='f8')

    domainload = dcop.load(pos, gamma=1)
    assert sum(domainload) == comm.allreduce(len(pos))

@MPITest(commsize=4)
def test_loadbalance(comm):
    DomainGrid = [[0, 1, 2, 3, 4], [0, 2, 4]]

    dcop = domain.GridND(DomainGrid,
            comm=comm,
            periodic=True)

    domainload = [5, 4, 9, 3, 15, 6, 8, 1]

    dcop.loadbalance(domainload)

    assert not any(dcop.DomainAssign - [3, 2, 1, 1, 0, 3, 2, 3])

@MPITest(commsize=4)
def test_loadbalance_degenerate(comm):
    DomainGrid = [[0, 1, 2, 3], [0, 3]]

    dcop = domain.GridND(DomainGrid,
            comm=comm,
            periodic=True)

    domainload = [10, 6, 12]

    dcop.loadbalance(domainload)

    assert not any(dcop.DomainAssign - [0, 1, 2])
