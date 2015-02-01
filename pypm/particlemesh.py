"""
    PyPM:
        A Particle Mesh code in Python

"""
import pfft
import domain
import cic
import numpy
from mpi4py import MPI

class ParticleMesh(object):
    def __init__(self, BoxSize, Nmesh, comm=None, np=None, verbose=False):
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
                np = pfft.split_size_2d(self.comm.size)

        self.procmesh = pfft.ProcMesh(np, comm=comm)
        self.Nmesh = Nmesh
        self.BoxSize = BoxSize
        self.partition = pfft.Partition(pfft.Type.PFFT_R2C, 
            [Nmesh, Nmesh, Nmesh], 
            self.procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_DESTROY_INPUT)

        self.real = pfft.LocalBuffer(self.partition).view_input()
        self.complex = pfft.LocalBuffer(self.partition).view_output()

        self.forward = pfft.Plan(self.partition, pfft.Direction.PFFT_FORWARD,
                self.real.base, self.complex.base, pfft.Type.PFFT_R2C, pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_DESTROY_INPUT)
        self.backward = pfft.Plan(self.partition, pfft.Direction.PFFT_BACKWARD,
                self.complex.base, self.real.base, pfft.Type.PFFT_C2R, pfft.Flags.PFFT_TRANSPOSED_IN | pfft.Flags.PFFT_DESTROY_INPUT)

        self.domain = domain.GridND(self.partition.i_edges)
        self.verbose = verbose
        self.stack = []

    def transform(self, x):
        ret = (1.0 * self.Nmesh / self.BoxSize) * x - self.partition.local_i_start
        #print self.partition.local_i_start, (ret.max(axis=0), ret.min(axis=0))
        return ret
    def transform0(self, x):
        ret = (1.0 * self.Nmesh / self.BoxSize) * x
        #print self.partition.local_i_start, (ret.max(axis=0), ret.min(axis=0))
        return ret

    def decompose(self, pos):
        """ create a domain decompose layout for particles at given mesh
            coordinates.

            position is in BoxSize coordinates (global)

            This layout can be used to migrate particles and images
            to the correct MPI ranks that hosts the PM local mesh

        """
        return self.domain.decompose(pos, smoothing=1.0,
                transform=self.transform0)

    def paint(self, pos, mass=1.0):
        """ 
            position is in BoxSize coordinates (global position)
            transform the particle field given by pos and mass
            to the overdensity field in fourier space and save
            it in the internal storage.
        """
        self.real[:] = 0
        cic.paint(pos, self.real, weights=mass, mode='ignore',
                period=self.Nmesh, transform=self.transform)

    def r2c(self, pos=None, mass=1.0):
        """ 
            position is in BoxSize coordinates (global position)

            transform the particle field given by pos and mass
            to the overdensity field in fourier space and save
            it in the internal storage.

            if pos is None, do not paint the paritlces, directly
            use self.real.
        """
        if pos is not None:
            self.paint(pos, mass)

        if self.verbose:
            realsum = self.comm.allreduce(self.real.sum(dtype='f8'), MPI.SUM)
            if self.comm.rank == 0:
                print 'before r2c, sum of real', realsum
            self.comm.barrier()

        self.forward.execute(self.real.base, self.complex.base)

        if self.procmesh.rank == 0:
            # remove the mean !
            # self.complex.flat[0] = 0
            pass

    def push(self):
        """ back up the old complex field """
        self.stack.append(self.complex.copy())

    def pop(self):
        """ restore the last backed up complex field """
        self.complex[:] = self.stack.pop()

    def transfer(self, *transfer_functions):
        """ apply a chain of transfer functions to the complex field
            this will destroy the complex field. (self.complex)
            There is no way back. save the complex field with push()
        """
        w = []
        for d in range(self.partition.Ndim):
            s = numpy.ones(self.partition.Ndim, dtype='intp')
            s[d] = self.partition.local_no[d]
            wi = numpy.arange(s[d], dtype='f8') + self.partition.local_o_start[d] 
            wi[wi >= self.Nmesh // 2] -= self.Nmesh
            wi *= (2 * numpy.pi / self.Nmesh)
            w.append(wi.reshape(s))

        for transfer in transfer_functions:
            if transfer.func_code.co_argcount == 2:
                transfer(self.complex, w)
            elif transfer.func_code.co_argcount == 3:
                transfer(self.comm, self.complex, w)
            else:
                raise TypeError(
                "Wrong definition of the transfer function: %s" % transfer.__name__)

    def c2r(self, pos, *transfer_functions):
        """ 
            pos is in BoxSize units (global position)

            complex is the fourier space field after applying the transfer 
            kernel.

            transfer is a callable:
            transfer(complex, w):  
            apply transfer function on complex field with given k:

            w is a tuple of (w0, w1, w2, ...)
            w is in circular frequency units. The dimensionful k is w * Nmesh / BoxSize 
            (nyquist is at about w = pi)
            they broadcast to the correct shape of complex. This is to reduce
            memory usage somewhat.
            complex is modified in place.
        """
        self.push()

        self.transfer(*transfer_functions)

        self.backward.execute(self.complex.base, self.real.base)
        if self.verbose:
            realsum = self.comm.allreduce(self.real.sum(dtype='f8'), MPI.SUM)
            if self.comm.rank == 0:
                print 'after c2r, sum of real', realsum
            self.comm.barrier()

        # restore the complex field, for next c2r transform
        self.pop()

        rt = cic.readout(self.real, pos, mode='ignore', period=self.Nmesh,
                transform=self.transform)
        return rt
