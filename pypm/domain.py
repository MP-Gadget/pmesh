"""
   Domain Decomposition in Gaepsi

    currently we have a GridND decomposition algorithm.

"""
from mpi4py import MPI
import numpy

class Rotator(object):
    def __init__(self, comm):
        self.comm = comm
    def __enter__(self):
        self.comm.Barrier()
        for i in range(self.comm.rank):
            self.comm.Barrier()
    def __exit__(self, type, value, tb):
        for i in range(self.comm.rank, self.comm.size):
            self.comm.Barrier()
        self.comm.Barrier()
def bincountv(x, weights, minlength=None, dtype=None):
    """ bincount with vector weights """
    weights = numpy.array(weights)
    if minlength == None:
        if len(x) == 0:
            minlength = 0
        else:
            minlength = x.max() + 1
    if dtype is None:
        dtype = weights.dtype

    shape = [minlength] + list(weights.shape[1:])

    out = numpy.empty(shape, dtype=dtype)
    for index in numpy.ndindex(*shape[1:]):
        ind = tuple([Ellipsis] + list(index))
        out[ind] = numpy.bincount(x, weights[ind], minlength=minlength)
    return out

class Layout(object):
    """ A global all to all communication layout 
        
    """
    def __init__(self, comm, oldlength, sendcounts, indices, recvcounts=None):
        """
        sendcounts is the number of items to send
        indices is the indices of the items in the data array.
        """

        self.comm = comm
        assert self.comm.size == sendcounts.shape[0]

        self.sendcounts = numpy.array(sendcounts, order='C')
        self.recvcounts = numpy.empty_like(self.sendcounts, order='C')

        self.sendoffsets = numpy.zeros_like(self.sendcounts, order='C')
        self.recvoffsets = numpy.zeros_like(self.recvcounts, order='C')

        if recvcounts is None:
            # calculate the recv counts array
            # ! Alltoall
            self.comm.Barrier()
            self.comm.Alltoall(self.sendcounts, self.recvcounts)
            self.comm.Barrier()
        else:
            self.recvcounts = recvcounts
        self.sendoffsets[1:] = self.sendcounts.cumsum()[:-1]
        self.recvoffsets[1:] = self.recvcounts.cumsum()[:-1]

        self.oldlength = oldlength
        self.newlength = self.recvcounts.sum()

        self.indices = indices

    def exchange(self, data):
        """ exchange the data globally according to the layout;

            data shall be of the same length of the input position
            that builds the layout

        """
        if len(data) != self.oldlength:
            raise ValueError(
            'the length of data does not match that used to build the layout')
        # lets check the data type first
        dtypes = self.comm.allgather(data.dtype.str)
        if len(set(dtypes)) != 1:
            raise TypeError('dtype of input differ on different ranks. %s' %
                    str(dtypes))

        #build buffer
        # Watch out: 
        # take produces C-contiguous array, 
        # friendly to alltoallv.
        # fancy indexing does not always return C_contiguous
        # array (2 days to realize this!)
        
        buffer = data.take(self.indices, axis=0)

        newshape = list(data.shape)
        newshape[0] = self.newlength

        # build a dtype for communication
        # this is to avoid 2GB limit from bytes.
        duplicity = numpy.product(numpy.array(data.shape[1:], 'intp')) 
        itemsize = duplicity * data.dtype.itemsize
        dt = MPI.BYTE.Create_contiguous(itemsize)
        dt.Commit()

        recvbuffer = numpy.empty(newshape, dtype=data.dtype, order='C')
        self.comm.Barrier()

        # now fire
        rt = self.comm.Alltoallv((buffer, (self.sendcounts, self.sendoffsets), dt), 
                            (recvbuffer, (self.recvcounts, self.recvoffsets), dt))
        dt.Free()
        self.comm.Barrier()
        return recvbuffer

    def gather(self, data, mode='sum'):
        """ 
            pull the data from other ranks back to its original hosting rank
            values of mirror items are added. 
            mode can be 'sum' or 'any', or 'mean'. 
        """
        # lets check the data type first
        if mode not in ['sum', 'any', 'mean']:
            raise ValueError('mode has to be "sum" or "any", "mean"')

        if len(data) != self.newlength:
            raise ValueError(
            'the length of data does not match result of a domain.exchange')

        dtypes = self.comm.allgather(data.dtype.str)
        if len(set(dtypes)) != 1:
            raise TypeError('dtype of input differ on different ranks. %s' %
                    str(dtypes))


        newshape = list(data.shape)
        newshape[0] = len(self.indices)

        # build a dtype for communication
        # this is to avoid 2GB limit from bytes.
        duplicity = numpy.product(numpy.array(data.shape[1:], 'intp')) 
        itemsize = duplicity * data.dtype.itemsize
        dt = MPI.BYTE.Create_contiguous(itemsize)
        dt.Commit()

        recvbuffer = numpy.empty(newshape, dtype=data.dtype, order='C')
        self.comm.Barrier()

        # now fire
        rt = self.comm.Alltoallv((data, (self.recvcounts, self.recvoffsets), dt), 
                            (recvbuffer, (self.sendcounts, self.sendoffsets), dt))
        dt.Free()
        self.comm.Barrier()

        if self.oldlength == 0:
            newshape[0] = 0
            return numpy.empty(newshape, data.dtype)

        if mode == 'sum':
            return bincountv(self.indices, recvbuffer, minlength=self.oldlength)
        if mode == 'mean':
            N = numpy.bincount(self.indices, minlength=self.oldlength)
            s = [self.oldlength] + [-1] * (len(newshape) - 1)
            N = N.reshape(s)
            return \
                    bincountv(self.indices, recvbuffer, minlength=self.oldlength) / N
        elif mode == 'any':
            mask = numpy.empty(len(self.indices), dtype='?')
            if len(mask) > 0:
                mask[0] = True
                mask[1:] = self.indices[1:] != self.indices[:-1]
            return recvbuffer[mask]

class GridND(object):
    """
        ND domain decomposition on a uniform grid
    """
    
    from _domain import gridnd_fill as _fill
    _fill = staticmethod(_fill)
    @staticmethod
    def _digitize(data, bins):
        if len(data) == 0:
            return numpy.empty((0), dtype='intp')
        else:
            return numpy.digitize(data, bins)

    def __init__(self, 
            grid,
            comm=MPI.COMM_WORLD,
            periodic=True):
        """ 
            grid is a list of  grid edges. 
            grid[0] or pos[:, 0], etc.
        
            grid[i][-1] are the boxsizes
            the ranks are set up into a mesh of 
                len(grid[0]) - 1, ...
        """
        self.dims = numpy.array([len(g) - 1 for g in grid], dtype='int32')
        self.grid = numpy.asarray(grid)
        self.periodic = periodic
        self.comm = comm
        assert comm.size == numpy.product(self.dims)
        rank = numpy.unravel_index(self.comm.rank, self.dims)

        self.myrank = numpy.array(rank)
        self.mystart = numpy.array([g[r] for g, r in zip(grid, rank)])
        self.myend = numpy.array([g[r + 1] for g, r in zip(grid, rank)])

    def decompose(self, pos, smoothing=0, transform=None):
        """ decompose the domain according to pos,

            smoothing is the size of a particle:
                any particle that intersects the domain will
                be transported to the domain.
            transform is the transformation on pos
            transform(pos[:, 3]) -> newpos[:, 3]
            returns a Layout object that can be used
            to exchange data
        """

        # we can't deal with too many points per rank, by  MPI
        assert len(pos) < 1024 * 1024 * 1024 * 2
        pos = numpy.asarray(pos)

        Npoint = len(pos)
        Ndim = len(self.dims)
        counts = numpy.zeros(self.comm.size, dtype='int32')
        periodic = self.periodic

        if Npoint != 0:
            sil = numpy.empty((Ndim, Npoint), dtype='i2', order='C')
            sir = numpy.empty((Ndim, Npoint), dtype='i2', order='C')
            chunksize = 1024 * 48 
            for i in range(0, Npoint, chunksize):
                s = slice(i, i + chunksize)
                chunk = transform(pos[s])
                for j in range(Ndim):
                    dim = self.dims[j]
                    if periodic:
                        tmp = numpy.remainder(chunk[:, j], self.grid[j][-1])
                    else:
                        tmp = chunk[:, j]
                    sil[j, s] = self._digitize(tmp - smoothing, self.grid[j]) - 1
                    sir[j, s] = self._digitize(tmp + smoothing, self.grid[j])

            for j in range(Ndim):
                dim = self.dims[j]
                if not periodic:
                    numpy.clip(sil[j], 0, dim, out=sil[j])
                    numpy.clip(sir[j], 0, dim, out=sir[j])

            self._fill(0, counts, self.dims, sil, sir, periodic)

            # now lets build the indices array.
            indices = self._fill(1, counts, self.dims, sil, sir, periodic)
            indices = numpy.array(indices, copy=False)
        else:
            indices = numpy.empty(0, dtype='int32')

        # create the layout object
        layout = Layout(
                comm=self.comm,
                oldlength=Npoint,
                sendcounts=counts,
                indices=indices)

        return layout

