"""
    CoArray/MPI in Python

    This is a simple example implements the CoArray 1.0 standard:

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

    No methods are provided for the global shape, global ndims yet.

    Currently the co-ndims is limited to 1:

    This can be done either use MPI's Cart Comm, or do some simple encoding.
    Because MPI.Comm doesn't quite work with the life cycle of python objects,
    requiring explicit MPI.Comm.Free,
    reshaping the co-ndims will be difficult if we hold MPI Cart Comm objects
    inside coarrays; unless we don't care about correctness.

    Yu Feng <rainwoodman@gmail.com>
"""

import numpy

from mpi4py import MPI

class coaproxy(object):
    def __init__(self, coa, image):
        self.image = image
        self.comm = coa.__coameta__.comm

        # root node of a proxy
        self.index = Ellipsis
        self.parent = None

    @classmethod
    def fancyindex(cls, parent, index):
        self = object.__new__(cls)
        self.image = parent.image
        self.comm = parent.comm
        self.index = index
        self.parent = parent
        return self

    @property
    def indices(self):
        indices = []
        while self is not None:
            indices.append(self.index)
            self = self.parent
        return list(reversed(indices))

    def __getitem__(self, index):
        return coaproxy.fancyindex(self, index)

    def __repr__(self):
        return 'coaproxy:%d/%d %s' % (self.image, self.comm.size, self.indices)

    def __str__(self):
        return 'coaproxy:%d/%d %s' % (self.image, self.comm.size, self.indices)

class coameta(object):
    def __init__(self, comm, coarray):
        self.comm = comm
        self.operations = []
        # in the future get the buffer and create MPI.Win

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
        self.__coameta__.operations.append((index, proxy))

    def sync(self, images=None):
        if images is None:
            images = range(self.num_images)

        comm = self.__coameta__.comm
        requests = []

        sendactions = [[] for i in range(comm.size)]

        recvactions = []

        # build the list of operations to the targets side
        for index, proxy in self.__coameta__.operations:
            if proxy.image not in images: continue

            assert proxy.comm == comm # must be from the same communicator!

            sendactions[proxy.image].append((proxy.indices, comm.rank))
            recvactions.append((index, proxy.image))

        sendactions = sum(comm.alltoall(sendactions), [])

        for sendindices, senddest in sendactions:
            data = self
            for index in sendindices:
                data = data[index]
#            print('sending to', senddest, data)
            requests.append(comm.isend(data, dest=senddest))

        for recvindex, recvsource in recvactions:
#            print('receiving from ', recvsource, recvindex)

            # using a blocked send to avoid if sending and receving from the
            # same buffer; or if recvindex is discontinous

            self[recvindex] = comm.recv(source=recvsource)

            # we can probably find a fix to work with irecv, but why bother.

            # requests.append(comm.irecv(buf=self[recvindex], source=recvsource))

        MPI.Request.waitall(requests)


def test_coarray(comm):
    coa = coarray.zeros(comm, (8, 3), dtype='f8')

    coa[...] = coa.thisimage

    left = (coa.thisimage - 1) % coa.num_images
    right = (coa.thisimage + 1) % coa.num_images

    print(coa(left)[:3])

    print('left, right', coa.thisimage, left, right)

    coa[:3, :2] = coa(left)[:3, :2]
    coa[-3:, -2:] = coa(right)[-3:, -2:]

    coa.sync([left])
    print(coa)

    coa.sync([right])
    print(coa)

if __name__ == '__main__':
    test_coarray(MPI.COMM_WORLD)

