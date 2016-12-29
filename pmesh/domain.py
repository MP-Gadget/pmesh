"""
   Domain Decomposition in Gaepsi

    currently we have a GridND decomposition algorithm.

"""
from mpi4py import MPI
import numpy

def bincountv(x, weights, minlength=None, dtype=None, out=None):
    """ bincount with vector weights """
    weights = numpy.array(weights)
    if minlength == None:
        if len(x) == 0:
            minlength = 0
        else:
            minlength = x.max() + 1
    if dtype is None:
        dtype = weights.dtype

    # protect dtype
    dtype = numpy.dtype(dtype)

    shape = [minlength] + list(weights.shape[1:])

    if out is None:
        out = numpy.empty(shape, dtype=dtype)

    for index in numpy.ndindex(*shape[1:]):
        ind = tuple([Ellipsis] + list(index))
        out[ind] = numpy.bincount(x, weights[ind], minlength=minlength)
    return out

def promote(data, comm):
    data = numpy.asarray(data)
    dtypes = comm.allgather(data.dtype.str)
    dtype = numpy.find_common_type(list(set(dtypes)), [])
    return numpy.asarray(data, dtype)

class Layout(object):
    """ 
    The communication layout of a domain decomposition.

    Given a set of particles, Layout object knows how to move particles around.
    Do not create a Layout object directly. Always use :py:meth:`GridND.decompose`.

    Useful methods are :py:meth:`exchange`, and :py:meth:`gather`.

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
        """ 
        Deliever data to the intersecting domains.

        Parameters
        ----------
        data     : array_like (, extra_dimensions)
            The data to be delievered. It shall be of the same length and of 
            the same ordering of the input position that builds the layout.
            Each element is a matching element of the position used in the call
            to :py:meth:`GridND.decompose`

        Returns
        -------
        newdata  : array_like 
            The data delievered to the domain.
            Ghosts are created if a particle intersects multiple domains.
            Refer to :py:meth:`gather` for collecting data of ghosts.

        """
        # first convert to array
        data = promote(data, self.comm)

        if any(self.comm.allgather(len(data) != self.oldlength)):
            raise ValueError(
            'the length of data does not match that used to build the layout')

        #build buffer
        # Watch out: 
        # take produces C-contiguous array, 
        # friendly to alltoallv.
        # fancy indexing does not always return C_contiguous
        # array (2 days to realize this!)
        
        buffer = data.take(self.indices, axis=0)

        # build a dtype for communication
        # this is to avoid 2GB limit from bytes.
        duplicity = numpy.product(numpy.array(data.shape[1:], 'intp')) 
        itemsize = duplicity * data.dtype.itemsize
        dt = MPI.BYTE.Create_contiguous(itemsize)
        dt.Commit()
        dtype = numpy.dtype((data.dtype, data.shape[1:]))
        recvbuffer = numpy.empty(self.newlength, dtype=dtype, order='C')
        self.comm.Barrier()

        # now fire
        rt = self.comm.Alltoallv((buffer, (self.sendcounts, self.sendoffsets), dt), 
                            (recvbuffer, (self.recvcounts, self.recvoffsets), dt))
        dt.Free()
        self.comm.Barrier()
        return recvbuffer

    def gather(self, data, mode='sum', out=None):
        """ 
        Pull the data from other ranks back to its original hosting rank.

        Attributes
        ----------
        data    :   array_like
            data for each received particles. 
            
        mode    : string 'sum', 'any', 'mean', 'all', 'min'
            :code:`all` is to return all ghosts without any reduction
            :code:`sum` is to add the ghosts together
            :code:`any` is to pick value of any ghosts
            :code:`min` is to pick value of any ghosts
            :code:`mean` is to use the mean of all ghosts

        out : array_like or None
            stores the result of the gather operation.

        Returns
        -------
        gathered  : array_like
            gathered data. It is of the same length and ordering of the original
            positions used in :py:meth:`domain.decompose`. When mode is 'all', 
            all gathered particles (corresponding to self.indices) are returned.
        
        """
        data = promote(data, self.comm)
        # lets check the data type first

        if any(self.comm.allgather(len(data) != self.newlength)):
            raise ValueError(
            'the length of data does not match result of a domain.exchange')

        # build a dtype for communication
        # this is to avoid 2GB limit from bytes.
        duplicity = numpy.product(numpy.array(data.shape[1:], 'intp')) 
        itemsize = duplicity * data.dtype.itemsize
        dt = MPI.BYTE.Create_contiguous(itemsize)
        dt.Commit()
        dtype = numpy.dtype((data.dtype, data.shape[1:]))

        recvbuffer = numpy.empty(len(self.indices), dtype=dtype, order='C')
        self.comm.Barrier()


        # now fire
        rt = self.comm.Alltoallv((data, (self.recvcounts, self.recvoffsets), dt), 
                            (recvbuffer, (self.sendcounts, self.sendoffsets), dt))
        dt.Free()
        self.comm.Barrier()

        if self.oldlength == 0:
            if out is None:
                out = numpy.empty(self.oldlength, dtype=dtype)
            return out

        if mode == 'all':
            if out is None:
                out = recvbuffer
            else:
                out[...] = recvbuffer
            return out
        if mode == 'sum':
            return bincountv(self.indices, recvbuffer, minlength=self.oldlength, out=out)

        if isinstance(mode, numpy.ufunc):
            arg = self.indices.argsort()
            recvbuffer = recvbuffer[arg]
            N = numpy.bincount(self.indices, minlength=self.oldlength)
            offset = numpy.zeros(self.oldlength, 'intp')
            offset[1:] = numpy.cumsum(N)[:-1]
            return mode.reduceat(recvbuffer, offset, out=out)

        if mode == 'mean':
            N = numpy.bincount(self.indices, minlength=self.oldlength)
            s = [self.oldlength] + [1] * (len(recvbuffer.shape) - 1)
            N = N.reshape(s)
            out = bincountv(self.indices, recvbuffer, minlength=self.oldlength, out=out)
            out[...] /= N
            return out

        if mode == 'any':
            if out is None:
                out = numpy.zeros(self.oldlength, dtype=dtype)
            out[self.indices] = recvbuffer
            return out
        raise NotImplementedError

class GridND(object):
    """
    GridND is domain decomposition on a uniform grid of N dimensions.

    The total number of domains is prod([ len(dir) - 1 for dir in edges]).

    Attributes
    ----------
    edges   : list  (Ndim)
        A list of edges of the edges in each dimension.
        edges[i] is the edges on direction i. edges[i] includes 0 and BoxSize.
    comm   : :py:class:`MPI.Comm`
        MPI Communicator, default is :code:`MPI.COMM_WORLD` 
 
    periodic : boolean
        Is the domain decomposition periodic? If so , edges[i][-1] is the period.


    """
    
    from ._domain import gridnd_fill as _fill
    _fill = staticmethod(_fill)
    @staticmethod
    def _digitize(data, bins):
        if len(data) == 0:
            return numpy.empty((0), dtype='intp')
        else:
            return numpy.digitize(data, bins)

    @classmethod
    def uniform(cls, BoxSize, comm=MPI.COMM_WORLD, periodic=True):
        ndim = len(BoxSize)

        # compute a optimal shape where each domain is as cubical as possible

        r = (1.0 * comm.size / numpy.prod(BoxSize) * min(BoxSize)) ** (1.0 / ndim)
        shape = [ r * (BoxSize[i] / min(BoxSize)) for i in range(ndim)]
        shape = numpy.array(shape)
        imax = shape.argmax()
        shape = numpy.int32(shape)
        shape[shape < 1] = 1
        shape[imax] = 1
        shape[imax] = comm.size // numpy.prod(shape)
        assert numpy.prod(shape) <= comm.size

        edges = []
        for i in range(ndim):
            edges.append(numpy.linspace(0, BoxSize[i], shape[i] + 1, endpoint=True))
        return cls(edges, comm, periodic)

    def __init__(self, 
            edges,
            comm=MPI.COMM_WORLD,
            periodic=True):
        """ 
        """
        self.shape = numpy.array([len(g) - 1 for g in edges], dtype='int32')
        self.ndim = len(self.shape)
        self.edges = numpy.asarray(edges)
        self.periodic = periodic
        self.comm = comm
        assert comm.size >= numpy.product(self.shape)

        # the following variables are not always defined if there are
        # more ranks than domains.

        #rank = numpy.unravel_index(self.comm.rank, self.shape)

        #self.myrank = numpy.array(rank)
        #self.mystart = numpy.array([g[r] for g, r in zip(edges, rank)])
        #self.myend = numpy.array([g[r + 1] for g, r in zip(edges, rank)])

    def decompose(self, pos, smoothing=0, transform=None):
        """ 
        Decompose particles into domains.

        Create a decomposition layout for particles at :code:`pos`.
        
        Parameters
        ----------
        pos       :  array_like (, ndim)
            position of particles, ndim can be more than the dimenions
            of the domains, in which case only the first few directions are used.

        smoothing : float
            Smoothing of particles. Any particle that intersects a domain will
            be transported to the domain. Smoothing is in the coordinate system
            of the edges.

        transform : callable
            Apply the transformation on pos before the decompostion.
            transform(pos[:, 3]) -> domain_pos[:, 3]
            transform is needed if pos and the domain edges are of different units.
            For example, pos in physical simulation units and domain edges on a mesh unit.

        Returns
        -------
        layout :  :py:class:`Layout` object that can be used to exchange data

        """

        # we can't deal with too many points per rank, by  MPI
        assert len(pos) < 1024 * 1024 * 1024 * 2
        pos = numpy.asarray(pos)

        assert pos.shape[1] >= self.ndim

        if transform is None:
            transform = lambda x: x
        Npoint = len(pos)
        counts = numpy.zeros(self.comm.size, dtype='int32')
        periodic = self.periodic

        if Npoint != 0:
            sil = numpy.empty((self.ndim, Npoint), dtype='i2', order='C')
            sir = numpy.empty((self.ndim, Npoint), dtype='i2', order='C')
            chunksize = 1024 * 48 
            for i in range(0, Npoint, chunksize):
                s = slice(i, i + chunksize)
                chunk = transform(pos[s])
                for j in range(self.ndim):
                    if periodic:
                        tmp = numpy.remainder(chunk[:, j], self.edges[j][-1])
                    else:
                        tmp = chunk[:, j]
                    sil[j, s] = self._digitize(tmp - smoothing, self.edges[j]) - 1
                    sir[j, s] = self._digitize(tmp + smoothing, self.edges[j])

            for j in range(self.ndim):
                dim = self.shape[j]
                if not periodic:
                    numpy.clip(sil[j], 0, dim, out=sil[j])
                    numpy.clip(sir[j], 0, dim, out=sir[j])

            self._fill(0, counts, self.shape, sil, sir, periodic)

            # now lets build the indices array.
            indices = self._fill(1, counts, self.shape, sil, sir, periodic)
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

