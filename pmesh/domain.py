"""
    Domain Decomposition in Gaepsi

    currently we have a GridND decomposition algorithm.

    A few concepts:

    - `local` and `ghost`. A data entry is local if
      it is stored on the current rank. A data entry is ghost
      if it is stored on another rank. 

    - `primary` and `padding`. A spatial position is
      primary if it is within the spatial boundary the current rank.
      If it is not, then it is in the `padding` region, which is
      determined by the `smoothing` parameter of decompose. If it
      is further out, there must have been an error.

    `local` and `primary` are not necessarily the same set of entries.

"""
from mpi4py import MPI
import numpy
from numpy.lib import recfunctions as rfn
import heapq

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
    dtype_root = comm.bcast(data.dtype)
    data = data.astype(dtype_root)
    shape_root = comm.bcast(data.shape)
    if shape_root[1:] != data.shape[1:]:
        raise ValueError('the shape of the data does not match across ranks.')
    return data

def pack_arrays(seq):
    """
    Pack a sequence of arrays to a structured array by copying the data.

    Unlike numpy's merge_arrays, this function preserves the shape of the columns
    rather than silently losing data.
    """
    dtype = []
    N = []
    for data in seq:
        data = numpy.asarray(data)
        dtype.append(('', (data.dtype, data.shape[1:])))
        N.append(data.shape[0])
    if not all(n == N[0] for n in N):
        raise ValueError('the shape of the data does not match across different columns.')

    dtype = numpy.dtype(dtype)
    out = numpy.empty(N[0], dtype=dtype)
    for key, data in zip(dtype.names, seq):
        data = numpy.asarray(data)
        out[key] = data
    return out

class Layout(object):
    """ 
    The communication layout of a domain decomposition.

    Given a set of particles, Layout object knows how to move particles around.
    Do not create a Layout object directly. Always use :py:meth:`GridND.decompose`.

    Useful methods are :py:meth:`exchange`, and :py:meth:`gather`.

    """
    def __init__(self, comm, sendlength, sendcounts, indices, recvcounts=None):
        """
        sendlength: the length of data to be sent.
        sendcounts: the number of items to send.
        recvcounts: if provided ust be an Alltoall transpose of sendcounts.
        indices: the indices for shuffling the data arrays during communication.
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

        self.sendlength = sendlength
        self.recvlength = self.recvcounts.sum()

        self.indices = indices

    def get_exchange_cost(self):
        """
        Returns an array of the exchange cost computed for each rank.

        The exchange cost of a rank is the sum of number of items sent from
        the rank to any rank that is not this rank.

        """
        mask = numpy.arange(self.comm.size) != self.comm.rank
        sendcount = numpy.sum(self.sendcounts[mask])
        allsendcount = self.comm.allgather(sendcount)
        return numpy.array(allsendcount)

    def exchange(self, *args, pack=True):
        """ 
        Delievers data to the intersecting domains.

        Parameters
        ----------
        *args : tuple of array_like (, extra_dimensions)
            data to be delievered are positional arguments.
            Every data item shall be of the same length and of the same
            ordering of the input position that builds the layout.
            Each element is a matching element of the position used
            in the call to :py:meth:`GridND.decompose`
        pack : boolean.  If True pack data entries into a structured array
            and make a single Alltoall exchange for all entries.

        Returns
        -------
        newdata  : tuple of array_like
            The data delievered to the domain.
            Ghosts are created if a particle intersects multiple domains.
            Refer to :py:meth:`gather` for collecting data of ghosts.

        """
        if pack:
            data = pack_arrays(args)
            newdata = self._exchange(data)
            r = tuple([newdata[name] for name in newdata.dtype.names])
        else:
            r = tuple([self._exchange(arg) for arg in args])
        if len(args) == 0:
            return None
        if len(args) == 1:
            return r[0]
        return r

    def _exchange(self, data):
        # first convert to array
        data = promote(data, self.comm)

        if any(self.comm.allgather(len(data) != self.sendlength)):
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
        recvbuffer = numpy.empty(self.recvlength, dtype=dtype, order='C')
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

        mode    : string 'sum', 'any', 'mean', 'all', 'local', or any numpy ufunc.
            :code:`'all'` is to return all results, local and ghosts, without any reduction
            :code:`'sum'` is to reduce the ghosts to the local with sum.
            :code:`'local'` is to remove ghosts, keeping only local
            :code:`'any'` is to reduce the ghosts to the local, by using any one local or ghost
            :code:`'mean'` is to reduce the ghosts, to use the mean (sum divided by the total number)
            any :code:`numpy.ufunc` is to reduce the ghosts, with the ufunc.
            `'sum'` is equivalent to `numpy.ufunc.add`

        out : array_like or None
            stores the result of the gather operation.

        Returns
        -------
        gathered  : array_like
            gathered data. It is of the same length and ordering of the original
            positions used in :py:meth:`domain.decompose`. When mode is 'all', 
            all gathered particles (corresponding to self.indices) are returned.

        """
        # lets check the data type first
        data = promote(data, self.comm)

        if any(self.comm.allgather(len(data) != self.recvlength)):
            raise ValueError(
            'the length of data does not match result of a domain.exchange')

        dtype = numpy.dtype((data.dtype, data.shape[1:]))

        if mode == 'local':
            self.comm.Barrier()
            # drop all ghosts communication is not needed
            if out is None:
                out = numpy.empty(self.sendlength, dtype=dtype)

            # indices uses send offsets
            start2 = self.sendoffsets[self.comm.rank]
            size2 = self.sendcounts[self.comm.rank]
            end2 = start2 + size2
            ind = self.indices[start2:end2]

            # data uses send offsets
            start1 = self.recvoffsets[self.comm.rank]
            size1 = self.recvcounts[self.comm.rank]
            end1 = start1 + size1
            out[ind] = data[start1:end1]

            return out

        # build a dtype for communication
        # this is to avoid 2GB limit from bytes.
        duplicity = numpy.product(numpy.array(data.shape[1:], 'intp')) 
        itemsize = duplicity * data.dtype.itemsize
        dt = MPI.BYTE.Create_contiguous(itemsize)
        dt.Commit()

        recvbuffer = numpy.empty(len(self.indices), dtype=dtype, order='C')
        self.comm.Barrier()


        # now fire
        rt = self.comm.Alltoallv((data, (self.recvcounts, self.recvoffsets), dt), 
                            (recvbuffer, (self.sendcounts, self.sendoffsets), dt))
        dt.Free()
        self.comm.Barrier()

        if self.sendlength == 0:
            if out is None:
                out = numpy.empty(self.sendlength, dtype=dtype)
            return out

        if mode == 'all':
            if out is None:
                out = recvbuffer
            else:
                out[...] = recvbuffer
            return out
        if mode == 'sum':
            return bincountv(self.indices, recvbuffer, minlength=self.sendlength, out=out)

        if isinstance(mode, numpy.ufunc):
            arg = self.indices.argsort()
            recvbuffer = recvbuffer[arg]
            N = numpy.bincount(self.indices, minlength=self.sendlength)
            offset = numpy.zeros(self.sendlength, 'intp')
            offset[1:] = numpy.cumsum(N)[:-1]
            return mode.reduceat(recvbuffer, offset, out=out)

        if mode == 'mean':
            N = numpy.bincount(self.indices, minlength=self.sendlength)
            s = [self.sendlength] + [1] * (len(recvbuffer.shape) - 1)
            N = N.reshape(s)
            out = bincountv(self.indices, recvbuffer, minlength=self.sendlength, out=out)
            out[...] /= N
            return out

        if mode == 'any':
            if out is None:
                out = numpy.zeros(self.sendlength, dtype=dtype)
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
    def _digitize(data, bins, right=False):
        if len(data) == 0:
            return numpy.empty((0), dtype='intp')
        else:
            return numpy.digitize(data, bins, right)

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
            periodic=True,
            DomainAssign=None):
        """ 
        DomainAssign records each domain is assigned to which rank
        """
        self.shape = numpy.array([len(g) - 1 for g in edges], dtype='int32')
        self.ndim = len(self.shape)
        self.edges = numpy.asarray(edges)
        self.periodic = periodic
        self.comm = comm
        self.size = numpy.product(self.shape)

        if DomainAssign is None:
            if comm.size >= self.size:
                DomainAssign = numpy.array(range(self.size), dtype='int32')
            else:
                DomainAssign = numpy.empty(self.size, dtype='int32')
                for i in range(comm.size):
                    start = i * self.size // comm.size
                    end = (i + 1) * self.size // comm.size
                    DomainAssign[start:end] = i

        self.DomainAssign = DomainAssign

        dd = numpy.zeros(self.shape, dtype='int16')

        for i, edge in enumerate(edges):
            edge = numpy.array(edge)
            dd1 = edge[1:] == edge[:-1]
            dd1 = dd1.reshape([-1 if ii == i else 1 for ii in range(self.ndim)])
            dd[...] |= dd1

        self.DomainDegenerate = dd.ravel()

        self._update_primary_regions()

    def load(self, pos, transform=None, gamma=2):
        """
        Returns the load of each domain, assuming that the load is a power-law N^gamma, where N is the number of particles within it.

        Parameters
        ----------
        pos       :  array_like (, ndim)
            position of particles, ndim can be more than the dimenions
            of the domains, in which case only the first few directions are used.

        Returns
        -------
        domainload       :  array_like
            The load of each domain.
        """

        # FIXME: this function looks like decompose; might be able to merge them into one?
        
        pos = numpy.asarray(pos)

        assert pos.shape[1] >= self.ndim

        if transform is None:
            transform = lambda x: x
        Npoint = len(pos)
        periodic = self.periodic

        if Npoint != 0:
            sil = numpy.empty((self.ndim, Npoint), dtype='i2', order='C')
            chunksize = 1024 * 48 
            for i in range(0, Npoint, chunksize):
                s = slice(i, i + chunksize)
                chunk = transform(pos[s])
                for j in range(self.ndim):
                    if periodic:
                        tmp = numpy.remainder(chunk[:, j], self.edges[j][-1])
                    else:
                        tmp = chunk[:, j]
                    sil[j, s] = self._digitize(tmp, self.edges[j]) - 1

            if periodic:
                mode = 'raise' # periodic box, must be in 0 ~ self.shape due to remainder
            else:
                mode = 'clip' # non periodic box, assign particles outside edges to the edge domains.
                              # FIXME: perhaps better to raise here? it may be very unbalanced if
                              # the edge is very far off!

            particle_domain = numpy.ravel_multi_index(sil, self.shape, mode=mode)
            tmp = numpy.bincount(particle_domain, minlength=self.size)
        else:
            tmp = numpy.zeros(self.size)

        domainload = self.comm.allreduce(tmp, op=MPI.SUM)

        domainload = domainload ** gamma

        return domainload


    def loadbalance(self, domainload):
        """
        Balancing the load of different ranks given the load of each domain. 
        The result is recorded in self.DomainAssign.

        Parameters
        ----------
        domainload       :  array_like
            The load of each domain. Can be calculated from self.load()

        """

        if self.size <= self.comm.size:
            return

        domains = sorted([ (domainload[i], i)
                              for i in range(self.size)],
                            reverse=True)

        # initially every rank is empty
        processes = [(0, i) for i in range(self.comm.size)]

        heapq.heapify(processes)

        for dload, dindex in domains:
            pload, rank = heapq.heappop(processes)
            pload += dload
            self.DomainAssign[dindex] = rank
            heapq.heappush(processes, (pload, rank))

        #update the primary region after balancing the load
        self._update_primary_regions()

    def _update_primary_regions(self):

        my_domains = numpy.where(self.DomainAssign == self.comm.rank)[0]

        N = len(my_domains)
        if N == 0:
            primary_region = None
        else:
            primary_region = {}
            primary_region['start'] = numpy.empty((N, self.ndim))
            primary_region['end'] = numpy.empty((N, self.ndim))
            for i in range(N):
                domain_index = numpy.unravel_index(my_domains[i], self.shape, order='C')
                primary_region['start'][i] = numpy.array([g[r] for g, r in zip(self.edges, domain_index)])
                primary_region['end'][i]   = numpy.array([g[r + 1] for g, r in zip(self.edges, domain_index)])

        self.primary_region = primary_region

    def isprimary(self, pos, transform=None):
        """
        Returns a boolean, True if the position falls into the primary region
            of the current rank.

        Parameters
        ----------
        pos       :  array_like (, ndim)
            position of particles, ndim can be more than the dimenions
            of the domains, in which case only the first few directions are used.

        Returns
        -------
        isprimary:  array_like, boolean.  Whether the position falls into
            the primary region of the rank.

        """

        if self.primary_region is None:
            return numpy.zeros(len(pos), dtype='?')

        if transform is None:
            transform = lambda x: x

        r = numpy.zeros(len(pos), dtype='?')

        x0 = self.primary_region['start']
        x1 = self.primary_region['end']

        BoxSize = numpy.array([self.edges[j][-1] for j in range(self.ndim)])
        chunksize = 1024 * 48

        for i in range(0, len(pos), chunksize):
            s = slice(i, i + chunksize)
            chunk = transform(pos[s])[..., :self.ndim]
            if self.periodic:
                chunk = numpy.remainder(chunk,  BoxSize)
            # looping over all primary regions.
            for j in range(len(x0)):
                r[s] += ((chunk >= x0[j]) & (chunk < x1[j])).all(axis=-1)
        return r

    def decompose(self, pos, smoothing=0, transform=None):
        """ 
        Decompose particles into domains.

        Create a decomposition layout for particles at :code:`pos`.
        
        Parameters
        ----------
        pos       :  array_like (, ndim)
            position of particles, ndim can be more than the dimenions
            of the domains, in which case only the first few directions are used.

        smoothing : float, or array_like
            Smoothing of particles. Any particle that intersects a domain will
            be transported to the domain. Smoothing is in the coordinate system
            of the edges. if array_like, smoothing per dimension.

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

        _smoothing = smoothing
        smoothing = numpy.empty(self.ndim, dtype='f8')
        smoothing[:] = _smoothing

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
                    tmp = chunk[:, j]
                    if periodic:
                        boxsize = self.edges[j][-1]
                        c = tmp % boxsize
                        l = self._digitize((c - smoothing[j]) % boxsize, self.edges[j], right=False)
                        r = self._digitize((c + smoothing[j]) % boxsize, self.edges[j], right=False)
                        p = self._digitize(c, self.edges[j], right=False)
                        l = p - (p - l) % self.shape[j] - 1
                        r = p + (r - p) % self.shape[j]
                        #print(l, p, r)
                        sil[j, s] = l
                        sir[j, s] = r
                    else:
                        l = self._digitize(tmp - smoothing[j], self.edges[j], right=False)
                        r = self._digitize(tmp + smoothing[j], self.edges[j], right=False)

                        sil[j, s] = (l - 1).clip(0, self.shape[j])
                        sir[j, s] = r.clip(0, self.shape[j])

#            for i in range(Npoint):
#                print(pos[i], smoothing, sil[..., i], sir[..., i])


            self._fill(0, counts, self.shape, sil, sir, periodic, self.DomainDegenerate, self.DomainAssign)

            # now lets build the indices array.
            indices = self._fill(1, counts, self.shape, sil, sir, periodic, self.DomainDegenerate, self.DomainAssign)

            indices = numpy.array(indices, copy=False)
        else:
            indices = numpy.empty(0, dtype='int32')

        # create the layout object
        layout = Layout(
                comm=self.comm,
                sendlength=Npoint,
                sendcounts=counts,
                indices=indices)

        return layout

