from mpi4py import MPI
import sys
import os.path
import traceback

d = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, d)
import numpy
from pypm import domain

fakecomm = lambda : None

fakecomm.size = 9
fakecomm.Alltoall = lambda a, b: None
fakecomm.Barrier = lambda : None
grid = [
        [0, 3, 6, 9] for dir in [0, 1]
        ]
pos = numpy.array(list(numpy.ndindex((10, 10))))
fakecomm.rank = 0

def inspect(layout):
    art = numpy.zeros((10, 10, 2), 'c1')
    for t, chunk in enumerate(numpy.split(
                pos[layout.indices], 
                layout.sendcounts.cumsum()[0:fakecomm.size - 1], 
                axis=0)):
        if len(chunk) == 0: continue
        art.fill('x')
        for i, j in numpy.ndindex(10, 10):
            rank = (i //3 ) * 3 + j // 3
            if i // 3 >= 3: continue
            if j // 3 >= 3: continue
            art[i, j, 0] = '%d' % rank

        for p in pos:
            art[p[0], p[1], 1] = ' '
        print('sending to', t, 'counts', layout.sendcounts[t])
        for p in chunk:
            art[p[0], p[1], 1] = '%d' % t
        print(art.view(dtype='S2').reshape(10, 10))

def test0():
    """ 
    this is to test there are no duplicated sent into a process
    """
    grid = [[0, 10], [0, 10]]
    fakecomm = lambda : None

    fakecomm.rank = 0
    fakecomm.size = 1
    fakecomm.Alltoall = lambda a, b: None
    fakecomm.Barrier = lambda : None
    dcop = domain.GridND(grid, 
            comm=fakecomm,
            periodic=True)
    layout = dcop.decompose(pos, smoothing=1)
    assert len(layout.indices) == 100

    # now test with 2 processes
    # this found a bug in the bubble sort
    fakecomm = lambda : None

    fakecomm.rank = 0
    fakecomm.size = 2
    grid = [[0, 5, 10], [0, 10]]
    fakecomm.Alltoall = lambda a, b: None
    fakecomm.Barrier = lambda : None

    dcop = domain.GridND(grid, 
            comm=fakecomm,
            periodic=True)
    layout = dcop.decompose(pos, smoothing=2)
    assert layout.sendcounts[0] == 90
    assert layout.sendcounts[1] == 90

def test1():
    """ no smoothing, no periodic """
    dcop = domain.GridND(grid, 
            comm=fakecomm,
            periodic=False)
    layout = dcop.decompose(pos, smoothing=0)
    assert (layout.sendcounts == 9).all()

def test2():
    """ no smoothing, periodic """
    dcop = domain.GridND(grid, 
            comm=fakecomm,
            periodic=True)
    layout = dcop.decompose(pos, smoothing=0)
    #inspect(layout)
    assert (layout.sendcounts == [16, 12, 12, 12, 9, 9, 12, 9, 9]).all()

def test3():
    """ with smoothing, no periodic"""
    dcop = domain.GridND(grid, 
            comm=fakecomm,
            periodic=False)
    layout = dcop.decompose(pos, smoothing=1)
#    inspect(layout)
    assert (layout.sendcounts == [16, 20, 20, 20, 25, 25, 20, 25, 25]).all()

def test4():
    """ with smoothing, periodic"""
    dcop = domain.GridND(grid, 
            comm=fakecomm,
            periodic=True)
    layout = dcop.decompose(pos, smoothing=1)
    #inspect(layout)
    assert (layout.sendcounts == [36, 30, 36, 30, 25, 30, 36, 30, 36]).all()

def test5():
    """ empty pos """

    if MPI.COMM_WORLD.size != 9: return

    dcop = domain.GridND(grid, 
            comm=MPI.COMM_WORLD,
            periodic=True)
    pos = numpy.empty((0, 2), dtype='f4')
    data = numpy.empty((0, 4), dtype='f4')
    if dcop.comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((10, 10))), dtype='f4')[:1]
        data = numpy.ones((len(pos), 4), dtype='f4')
    if dcop.comm.rank == 4:
        pos = numpy.array(list(numpy.ndindex((10, 10))), dtype='f4')[9:10]
        data = numpy.ones((len(pos), 4), dtype='f4')
    layout = dcop.decompose(pos, smoothing=1)
    newdata = layout.exchange(data)
    newpos = layout.exchange(pos)
    oldpos = layout.gather(newpos, mode='any')
    olddata = layout.gather(newdata, mode='mean')
    for i in range(dcop.comm.size):
        dcop.comm.barrier()
        if dcop.comm.rank == i:
            print(dcop.comm.rank, 'pos', pos)
            print('oldpos', oldpos)
            print('data', data)
            print('olddata', olddata)
            #print 'indices', layout.indices, 'sc', layout.sendcounts
            print('newpos', newpos)
            print('newdata', newdata)
            print('-----')
    assert numpy.allclose(oldpos, pos)
    assert numpy.allclose(olddata, data)
    # I am still not sure what the correct output is so just dump them out.

def test_wrongdtype():
    """ empty pos """

    if MPI.COMM_WORLD.size != 9: return

    dcop = domain.GridND(grid, 
            comm=MPI.COMM_WORLD,
            periodic=True)
    pos = numpy.empty((0, 2), dtype='f4')
    data = numpy.empty((0, 4), dtype='f4')
    if dcop.comm.rank == 0:
        pos = numpy.array(list(numpy.ndindex((10, 10))), dtype='f8')[:1]
        data = numpy.ones((len(pos), 4), dtype='f4')
    if dcop.comm.rank == 4:
        pos = numpy.array(list(numpy.ndindex((10, 10))), dtype='f8')[9:10]
        data = numpy.ones((len(pos), 4), dtype='f4')
    layout = dcop.decompose(pos, smoothing=1)
    try:
        newpos = layout.exchange(pos)
        raise AssertionError
    except TypeError as e:
        print('Expected Exception', e)
