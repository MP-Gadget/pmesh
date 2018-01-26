"""
    CoArray/MPI in Python

    This is a failed attempt to implement the CoArray 1.0 standard:

        http://caf.rice.edu/documentation/index.html

    The implementation uses the MPI 1.0 subset, mostly because I am
    not quite familar with MPI.Win objects and how they interact
    with numpy. mpi4py also has a few test blacklists for versions
    of openmpi that looked a bit scary.

    The main entry object is coarray a subclass of numpy.ndarray,

    A coarray can be created via `coarray.zeros` or `coarray.fromarray`

    A sync operation must be invoked to issue data transport.
    The standard does not expect any data transfer before sync is called.

    Comparing to CoFortran, the use of [] and () is inverted,
    because in Python [] is for slicing.

    Fortran methods (thisimage, sync) are implemented as properties.

    The standard mentioned `rank`, which is `ndims`. co-rank becomes condims.

    default attributes from numpy refers to local properties.

    No methods are provided for the global shape, global ndims.

    What doesn't work:

    Currently the co-ndims is limited to 1:

    This can be done either use MPI's Cart Comm, or do some simple encoding.
    Because MPI.Comm doesn't quite work with the life cycle of python objects,
    requiring explicit MPI.Comm.Free,
    reshaping the co-ndims will be difficult if we hold MPI Cart Comm objects
    inside coarrays; unless we don't care about correctness.

    Broadcast and Gather are not supported.

    Treating the coarray dimension as an array dimension (co-transpose) doesn't
    work neither.

    Currently there is no way to notify the source rank which coarray to use
    for the send operation.

    We need to add all coarrays to a central repo to
    do that; and then may need to crawl a dependency graph. I think there will
    be a lot of trouble ahead.

    But things may be simplified if we make use of the fact that all operations
    are ran symmetrically.

    Yu Feng <rainwoodman@gmail.com>
"""

import numpy

from mpi4py import MPI

class coaproxy(object):
    def __init__(self, coa, coindex):
        self.coindex = coindex
        self.coa = coa
        self.comm = coa.__coameta__.comm
        self.__coameta__ = coa.__coameta__

        # root node of a proxy
        self.index = Ellipsis
        self.parent = None

    @classmethod
    def fancyindex(cls, parent, index):
        self = object.__new__(cls)
        self.coindex = parent.coindex
        self.coa = parent.coa
        self.comm = parent.comm
        self.index = index
        self.parent = parent
        self.__coameta__ = parent.__coameta__
        return self

    @property
    def indices(self):
        indices = []
        while self is not None:
            indices.append(self.index)
            self = self.parent
        return list(reversed(indices))

    @property
    def isgroup(self):
        return self.coindex is Ellipsis

    def __getitem__(self, index):
        return coaproxy.fancyindex(self, index)

    def __setitem__(self, index, value):
        proxy = self[index]

        if self.isgroup:
            self.__coameta__.operations.append(Scatter(proxy, value))
        else:
            self.__coameta__.operations.append(Push(proxy, value))

    def __repr__(self):
        return 'coaproxy:%d/%d %s' % (self.coindex, self.comm.size, self.indices)

    def __str__(self):
        return 'coaproxy:%d/%d %s' % (self.coindex, self.comm.size, self.indices)

class Op(object): pass

class Pull(Op):
    def __init__(self, coa, localindex, proxy):
        self.localindex = localindex
        self.coa = coa
        self.proxy = proxy
        self.buffer = numpy.copy(coa[localindex], order='C') # continuous
        self.done = False

    def start(self):
        comm = self.proxy.comm
    #    print('irecv into', self.proxy.coindex, self.buffer.data)
        return comm.Irecv(buf=self.buffer, source=self.proxy.coindex)

    def finish(self):
        self.coa[self.localindex] = self.buffer
        self.buffer = None
        self.done = True

class Push(Op):
    def __init__(self, proxy, value):
        self.proxy = proxy
        self.buffer = value
        self.done = False

    def start(self):
        comm = self.proxy.comm
    #    print('isend to ', self.proxy.coindex, self.buffer.data)
        return comm.Isend(self.buffer, dest=self.proxy.coindex)

    def finish(self):
        self.done = True

class coameta(object):
    def __init__(self, comm, coarray):
        self.comm = comm
        self.operations = []
        # in the future get the buffer and create MPI.Win

    def _solve(self, images):
        comm = self.comm

        sendactions = [[] for i in range(comm.size)]
        recvactions = [[] for i in range(comm.size)]

        # build the list of operations to the targets side
        for op in self.operations:
            proxy = op.proxy
            assert proxy.comm == comm # must be from the same communicator!

            if proxy.coindex not in images: continue

            if isinstance(op, Pull):
                sendactions[proxy.coindex].append((proxy.indices, comm.rank))

            if isinstance(op, Push):
                recvactions[proxy.coindex].append((proxy.indices, comm.rank))

        sendactions = sum(comm.alltoall(sendactions), [])
        recvactions = sum(comm.alltoall(recvactions), [])

        return sendactions, recvactions

    def _start_operations(self, images):
        comm = self.comm

        requests = []
        ops = []
        for op in self.operations:
            proxy = op.proxy
            assert proxy.comm == comm # must be from the same communicator!

            if proxy.coindex not in images: continue

            request = op.start()
            requests.append(request)
            ops.append(op)

        return requests, ops

class coarray(numpy.ndarray):
    def __new__(kls, comm, *args, **kwargs):
        self = super(coarray, cls).__new__(cls, *args, **kwargs)
        self.__coameta__ = coameta(comm, self)
        return self

    @classmethod
    def fromarray(kls, comm, array):
        self = array.view(kls)
        self.__coameta__ = coameta(comm, self)
        return self

    @classmethod
    def zeros(kls, comm, shape, dtype='f8'):
        local = numpy.zeros(shape, dtype)
        return kls.fromarray(comm, local)

    def __setitem__(self, index, obj):
        if isinstance(obj, coaproxy):
            return self._setitem_proxy(index, obj)
        else:
            return super(coarray, self).__setitem__(index, obj)

    def __array_finalize__(self, obj):
        if obj is None: return

        if hasattr(obj, '__coameta__'):
            self.__coameta__ = obj.__coameta__
        else:
            self.__coameta__ = coameta(MPI.COMM_SELF, self)

    def __repr__(self):
        return 'coarray:%d/%d ' % (self.thisimage, self.num_images) + repr(self.view(numpy.ndarray))

    def __str__(self):
        return 'coarray:%d/%d ' % (self.thisimage, self.num_images) + str(self.view(numpy.ndarray))

    def __call__(self, coindex):
        """ indexing the coarray dimensions """
        return self.getimage(coindex)

    @property
    def thisimage(self):
        return self.__coameta__.comm.rank

    @property
    def num_images(self):
        return self.__coameta__.comm.size

    @property
    def coshape(self):
        return (self.num_images,)

    @property
    def condims(self): # rank in coarray fortran
        return 1

    def getimage(self, coindex):
        return coaproxy(self, coindex)

    def _setitem_proxy(self, index, proxy):
        self.__coameta__.operations.append(Pull(self, index, proxy))

    def sync(self, images=None):
        if images is None:
            images = range(self.num_images)

        comm = self.__coameta__.comm

        sendactions, recvactions = self.__coameta__._solve(images)

        requests, ops = self.__coameta__._start_operations(images)

        for sendindices, senddest in sendactions:
            value = self
            for index in sendindices:
                value = value[index]

            print('sending to', senddest, value.data, value, self)
            comm.Send(value, dest=senddest)

        for recvindices, recvsource in recvactions:
            value = self
            if len(recvindices) == 0:
                recvindices = recvindices + [Ellipsis]

            for index in recvindices[:-1]:
                value = value[index]

            print('receiving from ', recvsource, value.data)
            # trigger setitem
            buf = numpy.zeros_like(value[recvindices[-1]])
            comm.Recv(buf, recvsource)
            value[recvindices[-1]] = buf

        MPI.Request.waitall(requests)

        for op in ops: op.finish()

        self.__coameta__.operations = [op
                for op in self.__coameta__.operations
                if not op.done ]

def test_coarray(comm):
    coa = coarray.zeros(comm, (8, 3), dtype='f8')

    coa[...] = coa.thisimage

    left = (coa.thisimage - 1) % coa.num_images
    right = (coa.thisimage + 1) % coa.num_images

#    print(coa(left)[:3])

    print('left, right', coa.thisimage, left, right)

    coa[0] = coa(left)[0]
    coa[-1] = coa(right)[-1]

    coa.sync([left])
    assert (coa[0] == left).all()
    assert (coa[-1] == coa.thisimage).all()

    coa.sync([right])
    assert (coa[0] == left).all()
    assert (coa[-1] == right).all()

    coa(left)[1] = coa[1]
    coa(right)[-2] = coa[-2]

    coa.sync([left])
    assert (coa[1] == right).all()

    coa.sync([right])
    assert (coa[-2] == left).all()

#    coa(right)[3:4, :2] = coa[3:4, :2]

def test_cotranspose(comm):
    coa1 = coarray.zeros(comm, (comm.size, 3), dtype='f8')
    coa2 = coarray.zeros(comm, (comm.size, 3), dtype='f8')

    coa2[...] = coa2.thisimage

    for i in range(coa1.num_images):
        coa1[i] = coa2(i)[coa1.thisimage]

    coa1.sync()

    print(coa1)

if __name__ == '__main__':
#    test_coarray(MPI.COMM_WORLD)
    test_cotranspose(MPI.COMM_WORLD)

