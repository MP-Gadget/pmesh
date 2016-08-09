import numpy
import pfft
from . import domain
from . import cic

class RealField(numpy.ndarray):
    def __new__(kls, pm):
        buffer = pfft.LocalBuffer(pm.partition)
        self = buffer.view_input().view(type=kls)
        self.local_buffer = buffer
        self.pm = pm
        self.partition = pm.partition
        self.BoxSize = pm.BoxSize
        self.Nmesh = pm.Nmesh
        return self

    def copy(self):
        other = RealField(self.pm)
        other[...] = self
        return other

    def r2c(self, out):
        """ 
        Perform real to complex FFT on the internal canvas.

        The complex field will be dimensionless; this is to ensure if NormalizeDC
        is applyed, c2r produces :math:`1 + \delta` as expected.

        (To obtain CFT, multiply by :math:`L^3` from the :math:`dx^3` factor )

        Therefore, the zeroth component of the complex field is :math:`\\bar\\rho`.

        """

        assert isinstance(out, ComplexField)

        self.pm.forward.execute(self.local_buffer, out.local_buffer)

        # PFFT normalization, same as FastPM
        self[...] *= numpy.prod(self.pm.Nmesh ** -1.0)

    def paint(self, pos, mass=1.0, resample="cic", hold=False):
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
        pos    : array_like (, Ndim)
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
        def transform(self, x):
            ret = (1.0 * self.Nmesh / self.BoxSize) * x - self.partition.local_i_start
            return ret

        if not hold:
            self[...] = 0

        cic.paint(pos, self, weights=mass,
                            mode='ignore', period=self.Nmesh, transform=transform)

    def readout(self, pos, resample="cic"):
        """ 
        Read out from real field at positions

        Parameters
        ----------
        pos    : array_like (, Ndim)
            position of particles in simulation  unit

        Returns
        -------
        rt     : array_like (,)
            read out values from the real field.
 
        """
        # Transform from simulation unit to local grid unit.
        def transform(self, x):
            ret = (1.0 * self.Nmesh / self.BoxSize) * x - self.partition.local_i_start
            return ret

        rt = cic.readout(self, pos, mode='ignore', period=self.Nmesh,
                transform=transform)
        return rt

    def slabiter(self):
        """ returns a iterator of (x, y, z, ...), realfield """
        axissort = numpy.argsort(self.strides)[::-1]

        optimized = self.transpose(axissort)
        x = [pm.x[d].transpose(axissort) for d in range(len(self.shape))]

        for irow in self.shape[axissort[0]]: # iterator the slowest axis in memory
            kk = [x[d] if d != axissort[0] else x[d][irow] for d in range(len(self.shape))]
            yield kk, optimized[irow]


class ComplexField(numpy.ndarray):
    def __new__(kls, pm):
        buffer = pfft.LocalBuffer(pm.partition)
        self = buffer.view_output().view(type=kls)
        self.pm = pm
        self.partition = pm.partition
        self.local_buffer = buffer
        self.BoxSize = pm.BoxSize
        self.Nmesh = pm.Nmesh
        return self

    def c2r(self, out):
        assert isinstance(out, RealField)
        self.pm.backward.execute(self.local_buffer, out.local_buffer)

    def slabiter(self):
        """ returns a iterator of (kx, ky, kz), complexfield"""
        axissort = numpy.argsort(self.strides)[::-1]

        optimized = self.transpose(axissort)
        k = [pm.k[d].transpose(axissort) for d in range(len(self.shape))]

        for irow in self.shape[axissort[0]]: # iterator the slowest axis in memory
            kk = [k[d] if d != axissort[0] else k[d][irow] for d in range(len(self.shape))]
            yield kk, optimized[irow]

    def copy(self):
        other = ComplexField(self.pm)
        other[...] = self
        return other

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

        for d in range(self.partition.Ndim):
            t = numpy.ones(self.partition.Ndim, dtype='intp')
            s = numpy.ones(self.partition.Ndim, dtype='intp')
            t[d] = self.partition.local_ni[d]
            s[d] = self.partition.local_no[d]
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
        pos    : array_like (, Ndim)
            position of particles in simulation  unit

        Returns
        -------
        layout  : :py:class:domain.Layout
            layout that can be used to migrate particles and images
        to the correct MPI ranks that hosts the PM local mesh
        """

        # Transform from simulation unit to global grid unit.
        def transform0(self, x):
            ret = (1.0 * self.Nmesh / self.BoxSize) * x
            return ret

        return self.domain.decompose(pos, smoothing=1.0,
                transform=self.transform0)

