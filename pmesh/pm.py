import numpy
import pfft
import mpsort
from . import domain
from . import cic
from . import window

class Field(numpy.ndarray):
    """ Base class for RealField and ComplexField.

        It only supports those two subclasses.
    """
    def copy(self):
        other = self.__class__(self.pm)
        other[...] = self
        return other

    def add_attrs(self, buffer, pm):
        """ Used internally to add shortcuts of attributes from pm """
        self.pm = pm
        self.partition = pm.partition
        self.BoxSize = pm.BoxSize
        self.Nmesh = pm.Nmesh
        if isinstance(self, RealField):
            self.start = self.partition.local_i_start
            self.global_shape = pm.Nmesh
            self.x = pm.x
        else:
            self.start = self.partition.local_o_start
            self.global_shape = pm.Nmesh.copy()
            self.global_shape[-1] = self.global_shape[-1] // 2 + 1
            self.x = pm.k

        self.slices = tuple([
                slice(s, s + n)
                for s, n in zip(self.start, self.shape)
                ])

    def sort(self, out=None):
        """ Sort the field to a C_CONTIGUOUS array, partitioned by MPI ranks. """
        ind = numpy.ravel_multi_index(numpy.mgrid[self.slices], self.global_shape)
        if out is None:
            out = self
        return mpsort.sort(self.flat, orderby=ind.flat, comm=self.pm.comm, out=out.flat)

    def slabiter(self, index_type="coordinate"):
        """ returns a iterator of (x, y, z, ...), slab """

        # we iterate over the slowest axis to gain locality.
        axissort = numpy.argsort(self.strides)[::-1]
        optimized = self.transpose(axissort)
        if index_type == "coordinate":
            x = [self.x[d].transpose(axissort) for d in range(len(self.shape))]
        else:
            raise
        for irow in range(self.shape[axissort[0]]): # iterator the slowest axis in memory
            kk = [x[d][0] if d != axissort[0] else x[d][irow] for d in range(len(self.shape))]
            yield kk, optimized[irow]

class RealField(Field):
    methods = {
        'cic' : window.CIC,
        'tsc' : window.TSC,
        'cubic' : window.CUBIC,
        'lanczos2' : window.LANCZOS2,
        'lanczos3' : window.LANCZOS3,
    }

    def __new__(kls, pm):
        buffer = pfft.LocalBuffer(pm.partition)
        self = buffer.view_input(type=kls)
        Field.add_attrs(self, buffer, pm)
        return self

    def r2c(self, out):
        """ 
        Perform real to complex FFT on the internal canvas.

        The complex field will be dimensionless; this is to ensure if NormalizeDC
        is applyed, c2r produces :math:`1 + \delta` as expected.

        (To obtain CFT, multiply by :math:`L^3` from the :math:`dx^3` factor )

        Therefore, the zeroth component of the complex field is :math:`\\bar\\rho`.

        """

        assert isinstance(out, ComplexField)

        self.pm.forward.execute(self.base, out.base)

        # PFFT normalization, same as FastPM
        out[...] *= numpy.prod(self.pm.Nmesh ** -1.0)

    def paint(self, pos, mass=1.0, method="cic", hold=False):
        """ 
        Paint particles into the internal real canvas. 

        Transform the particle field given by pos and mass
        to the overdensity field in fourier space and save
        it in the internal storage. 
        A multi-linear CIC approximation scheme is used.

        The function can be called multiple times: 
        the result is cummulative. In a multi-step simulation where
        :py:class:`ParticleMesh` object is reused,  before calling 
        :py:meth:`paint`, make sure the canvas is cleared with :py:meth:`clear`.

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation unit

        mass   : scalar or array_like (,)
            mass of particles in simulation unit

        hold   : bool
            If true, do not clear the current value in the field.

        Notes
        -----
        the painter operation conserves the total mass. It is not the density.

        """
        # Transform from simulation unit to local grid unit.
        affine = window.Affine(self.ndim,
                    translate=-self.start,
                    scale=1.0 * self.Nmesh / self.BoxSize,
                    period = self.Nmesh)

        method = self.methods[method]

        if not hold:
            self[...] = 0

        method.paint(self, pos, mass, transform=affine)

    def readout(self, pos, out=None, method="cic"):
        """ 
        Read out from real field at positions

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation  unit

        Returns
        -------
        rt     : array_like (,)
            read out values from the real field.

        """
        # Transform from simulation unit to local grid unit.
        affine = window.Affine(self.ndim,
                    translate=-self.start,
                    scale=1.0 * self.Nmesh / self.BoxSize,
                    period = self.Nmesh)

        method = self.methods[method]

        return method.readout(self, pos, out=out, transform=affine)


class ComplexField(Field):
    def __new__(kls, pm):
        buffer = pfft.LocalBuffer(pm.partition)
        self = buffer.view_output(type=kls)
        Field.add_attrs(self, buffer, pm)
        return self

    def c2r(self, out):
        assert isinstance(out, RealField)
        self.pm.backward.execute(self.base, out.base)

    def resample(self, out):
        assert isinstance(out, ComplexField)

        tmp = numpy.empty_like(self)
        self.sort(out=tmp)

        # indtable stores the index in pmsrc for the mode in pmdest
        # since pmdest < pmsrc, all items are alright.
        indtable = [reindex(self.Nmesh[d], out.Nmesh[d]) for d in range(self.ndim)]

        ind = build_index(
                [t[numpy.r_[s]]
                for t, s in zip(indtable, out.slices) ],
                self.global_shape)

        # fill the points that has values in pmsrc
        mask = ind >= 0
        # their indices
        argind = ind[mask]
        # take the data
        data = mpsort.take(tmp.flat, argind, self.pm.comm)
        # fill in the value
        out[mask] = data

def build_index(indices, fullshape):
    """
        Build a linear index array based on indices on an array of fullshape.
        This is similar to numpy.ravel_multi_index.

        index value of -1 will on any axes will be translated to -1 in the final.

        Parameters:
            indices : a tuple of index per dimension.

            fullshape : a tuple of the shape of the full array

        Returns:
            ind : a 3-d array of the indices of the coordinates in indices in
                an array of size fullshape. -1 if any indices is -1.

    """
    localshape = [ len(i) for i in indices]
    ndim = len(localshape)
    ind = numpy.zeros(localshape, dtype='i8')
    for d in range(len(indices)):
        i = indices[d]
        i = i.reshape([-1 if dd == d else 1 for dd in range(ndim)])
        ind[...] *= fullshape[d]
        ind[...] += i

    mask = numpy.zeros(localshape, dtype='?')

    # now mask out bad points by -1
    for d in range(len(indices)):
        i = indices[d]
        i = i.reshape([-1 if dd == d else 1 for dd in range(ndim)])
        mask |= i == -1

    ind[mask] = -1
    return ind

def reindex(Nsrc, Ndest):
    """ returns the index in the frequency array for corresponding
        k in Nsrc and composes Ndest

        For those Ndest that doesn't exist in Nsrc, return -1

        Example:
        >>> reindex(8, 4)
        >>> array([0, 1, 2, 7])
        >>> reindex(4, 8)
        >>> array([ 0,  1,  2, -1, -1, -1,  -1,  3])

    """
    reindex = numpy.arange(Ndest)
    reindex[Ndest // 2 + 1:] = numpy.arange(Nsrc - Ndest // 2 + 1, Nsrc, 1)
    reindex[Nsrc // 2 + 1: Ndest -Nsrc //2 + 1] = -1
    return reindex

class ParticleMesh(object):
    """
    ParticleMesh provides an interface to solver for forces
    with particle mesh method

    ParticleMesh does not deal with memory. Use RealField(pm) and ComplexField(pm)
    to create memory buffers.

    Attributes
    ----------
    np      : array_like (npx, npy)
        The shape of the process mesh. This is the number of domains per direction.
        The product of the items shall equal to the size of communicator. 
        For example, for 64 rank job, np = (8, 8) is a good choice.
        Since for now only 3d simulations are supported, np must be of length-2.
        The default is try to split the total number of ranks equally. (eg, for
        a 64 rank job, default is (8, 8)

    comm    : :py:class:`MPI.Comm`
        the MPI communicator, (default is MPI.COMM_WORLD)

    Nmesh   : array of int
        number of mesh points per side.

    BoxSize : float
        size of box

    domain   : :py:class:`pmesh.domain.GridND`
        domain decomposition (private)

    partition : :py:class:`pfft.Partition`
        domain partition (private)

    w   : list
        a list of the circular frequencies along each direction (-pi to pi)
    k   : list
        a list of the wave numbers k along each direction (- pi N/ L to pi N/ L)
    x   : list
        a list of the position along each direction (-L/2 to L/ 2). x is conjugate of k.
    r   : list
        a list of the mesh position along each direction (-N/2 to N/2). r is conjugate of w.

    """

    def __init__(self, Nmesh, BoxSize=1.0, comm=None, np=None, dtype='f8', plan_method='estimate'):
        """ create a PM object.  """
        # this weird sequence to intialize comm is because
        # we want to be compatible with None comm == MPI.COMM_WORLD
        # while not relying on pfft's full mpi4py compatibility
        # (passing None through to pfft)
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        if np is None:
            if len(Nmesh) >= 3:
                np = pfft.split_size_2d(self.comm.size)
            else:
                np = [self.comm.size]

        dtype = numpy.dtype(dtype)
        if dtype is numpy.dtype('f8'):
            forward = pfft.Type.PFFT_R2C
            backward = pfft.Type.PFFT_C2R
        elif dtype is numpy.dtype('f4'):
            forward = pfft.Type.PFFTF_R2C
            backward = pfft.Type.PFFTF_C2R
        else:
            raise ValueError("dtype must be f8 or f4")

        self.procmesh = pfft.ProcMesh(np, comm=comm)
        self.Nmesh = numpy.array(Nmesh, dtype='i8')
        self.BoxSize = numpy.empty(len(Nmesh), dtype='f8')
        self.BoxSize[:] = BoxSize
        self.partition = pfft.Partition(forward,
            self.Nmesh,
            self.procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_PADDED_R2C)

        bufferin = pfft.LocalBuffer(self.partition)
        bufferout = pfft.LocalBuffer(self.partition)

        plan_method = {
            "estimate": pfft.Flags.PFFT_ESTIMATE,
            "measure": pfft.Flags.PFFT_MEASURE,
            "exhaustive": pfft.Flags.PFFT_EXHAUSTIVE,
            } [plan_method]

        self.forward = pfft.Plan(self.partition, pfft.Direction.PFFT_FORWARD,
                bufferin, bufferout, forward,
                plan_method | pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_TUNE | pfft.Flags.PFFT_PADDED_R2C)
        self.backward = pfft.Plan(self.partition, pfft.Direction.PFFT_BACKWARD,
                bufferout, bufferin, backward, 
                plan_method | pfft.Flags.PFFT_TRANSPOSED_IN | pfft.Flags.PFFT_TUNE | pfft.Flags.PFFT_PADDED_C2R)

        self.domain = domain.GridND(self.partition.i_edges, comm=self.comm)

        k = []
        x = []
        w = []
        r = []

        for d in range(self.partition.ndim):
            t = numpy.ones(self.partition.ndim, dtype='intp')
            s = numpy.ones(self.partition.ndim, dtype='intp')
            t[d] = self.partition.local_i_shape[d]
            s[d] = self.partition.local_o_shape[d]
            wi = numpy.arange(s[d], dtype='f4') + self.partition.local_o_start[d] 
            ri = numpy.arange(t[d], dtype='f4') + self.partition.local_i_start[d] 

            wi[wi >= self.Nmesh[d] // 2] -= self.Nmesh[d]
            ri[ri >= self.Nmesh[d] // 2] -= self.Nmesh[d]

            wi *= (2 * numpy.pi / self.Nmesh[d])
            ki = wi * self.Nmesh[d] / self.BoxSize[d]
            xi = ri * self.BoxSize[d] / self.Nmesh[d]

            w.append(wi.reshape(s))
            r.append(ri.reshape(t))
            k.append(ki.reshape(s))
            x.append(xi.reshape(t))

        self.w = w
        self.r = r
        self.k = k
        self.x = x

    def decompose(self, pos):
        """ 
        Create a domain decompose layout for particles at given
        coordinates.

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation  unit

        Returns
        -------
        layout  : :py:class:domain.Layout
            layout that can be used to migrate particles and images
        to the correct MPI ranks that hosts the PM local mesh
        """

        # Transform from simulation unit to global grid unit.
        def transform0(x):
            ret = (1.0 * self.Nmesh / self.BoxSize) * x
            return ret

        return self.domain.decompose(pos, smoothing=1.0,
                transform=transform0)

