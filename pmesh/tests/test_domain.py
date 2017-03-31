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
    assert nmass.dtype == numpy.dtype('complex128')
    nmass = comm.allgather(nmass)
    assert_array_equal(nmass[0], [0, 1])
    assert_array_equal(nmass[1], [2, 3])

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

    npos = comm.allgather(npos)
    assert_array_equal(npos[0], [[0, 0], [0, 1], [1, 0], [1, 1]])
    assert_array_equal(npos[1], [[0, 0], [0, 1], [1, 0], [1, 1]])

