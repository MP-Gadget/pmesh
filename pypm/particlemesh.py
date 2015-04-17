"""
PyPM: A Particle Mesh code in Python

"""
import pfft
import domain
import cic
import numpy
import time
from mpi4py import MPI
from tools import Timers

class ParticleMesh(object):
    """
    ParticleMesh provides an interface to solver for forces
    with particle mesh method

    ParticleMesh object is a state machine. 
    The typical routine is

    1. Paint particles via :py:meth:`paint`
    2. Real to Complex transform via :py:meth:`r2c`
    3. Complex to Real transform via :py:meth:`c2r`, applying transfer functions
    4. Read out force values     via :py:meth:`readout`
    5. go back to 3, for other force components

    Memory usage is twice the size of the FFT mesh. However, 
    Be aware the transfer functions may take more memory.

    Attributes
    ----------
    comm    : :py:class:`MPI.Comm`
        the MPI communicator

    Nmesh   : int
        number of mesh points per side

    BoxSize : float
        size of box
    
    domain   : :py:class:`pypm.domain.GridND`
        domain decomposition (private)

    partition : :py:class:`pfft.Partition`
        domain partition (private)

    real   : array_like
        the real FFT array (private)

    complex : array_like
        the complex FFT array (private)

    T    : :py:class:`pypm.tools.Timers`
        profiling timers

    """
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

        self.T = Timers(self.comm)
        with self.T['Plan']:
            self.forward = pfft.Plan(self.partition, pfft.Direction.PFFT_FORWARD,
                    self.real.base, self.complex.base, pfft.Type.PFFT_R2C,
                    pfft.Flags.PFFT_ESTIMATE | pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_DESTROY_INPUT)
            self.backward = pfft.Plan(self.partition, pfft.Direction.PFFT_BACKWARD,
                    self.complex.base, self.real.base, pfft.Type.PFFT_C2R, 
                    pfft.Flags.PFFT_ESTIMATE | pfft.Flags.PFFT_TRANSPOSED_IN | pfft.Flags.PFFT_DESTROY_INPUT)

        self.domain = domain.GridND(self.partition.i_edges)
        self.verbose = verbose
        self.stack = []

    def transform(self, x):
        """ 
        Transform from simulation unit to local grid unit.

        Parameters
        ----------
        x     : array_like (, Ndim)
            coordinates in simulation unit

        Returns
        -------
        ret   : array_like (, Ndim)
            coordinates in local grid unit
 
        """
        ret = (1.0 * self.Nmesh / self.BoxSize) * x - self.partition.local_i_start
        #print self.partition.local_i_start, (ret.max(axis=0), ret.min(axis=0))
        return ret

    def transform0(self, x):
        """ 
        Transform from simulation unit to global grid unit.

        Parameters
        ----------
        x     : array_like (, Ndim)
            coordinates in simulation unit

        Returns
        -------
        ret   : array_like (, Ndim)
            coordinates in global grid unit
 
        """
        ret = (1.0 * self.Nmesh / self.BoxSize) * x
        #print self.partition.local_i_start, (ret.max(axis=0), ret.min(axis=0))
        return ret

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
        with self.T['Decompose']:
            return self.domain.decompose(pos, smoothing=1.0,
                    transform=self.transform0)

    def paint(self, pos, mass=1.0):
        """ 
        Paint particles into the internal real canvas.

        Transform the particle field given by pos and mass
        to the overdensity field in fourier space and save
        it in the internal storage.
        A multi-linear CIC approximation scheme is used.

        Parameters
        ----------
        pos    : array_like (, Ndim)
            position of particles in simulation  unit

        mass   : scalar or array_like (,)
            mass of particles in simulation  unit


        Notes
        -----
        self.real is NOT the density field after this operation.
        It has the dimension of density, but is divided by 
        Nmesh **3; PFFT will adjust for this Nmesh**3 after r2c .

        """
        with self.T['Paint']:
            self.real[:] = 0
            cic.paint(pos, self.real, weights=mass, mode='ignore',
                    period=self.Nmesh, transform=self.transform)
            self.real *= (1.0 / self.BoxSize) ** 3

    def r2c(self, pos=None, mass=1.0):
        """ 
        Perform real to complex FFT on the internal canvas.

        If pos and mass are given, :py:meth:`paint` is called to
        paint the particles before running the fourier transform

        Parameters
        ----------
        pos    : array_like (, Ndim)
            position of particles in simulation  unit

        mass   : scalar or array_like (,)
            mass of particles in simulation  unit

        """
        if pos is not None:
            self.paint(pos, mass)

        if self.verbose:
            realsum = self.comm.allreduce(self.real.sum(dtype='f8'), MPI.SUM)
            if self.comm.rank == 0:
                print 'before r2c, sum of real', realsum
            self.comm.barrier()

        with self.T['R2C']:
            self.forward.execute(self.real.base, self.complex.base)

        if self.procmesh.rank == 0:
            # remove the mean !
            # self.complex.flat[0] = 0
            pass

    def push(self):
        """ 
        Save the complex field 
        
        The complex field is saved to an internal stack. Recover the
        complex field with :py:meth:`pop`.

        """
        self.stack.append(self.complex.copy())

    def pop(self):
        """ 
        Restore the complex field 
        
        The complex field was saved to an internal stack by :py:meth:`push`. 

        """
        self.complex[:] = self.stack.pop()

    def transfer(self, transfer_functions):
        """ 
        Apply transfer functions

        Apply a chain of transfer functions to the complex field, in place.
        If the original field shall be preserved, use :py:meth:`push`.
    
        Parameters
        ----------
        transfer_functions : list of :py:class:transfer.TransferFunction 
            A chain of transfer functions to apply to the complex field. 
        
        """
        w = []
        for d in range(self.partition.Ndim):
            s = numpy.ones(self.partition.Ndim, dtype='intp')
            s[d] = self.partition.local_no[d]
            wi = numpy.arange(s[d], dtype='f8') + self.partition.local_o_start[d] 
            wi[wi >= self.Nmesh // 2] -= self.Nmesh
            wi *= (2 * numpy.pi / self.Nmesh)
            w.append(wi.reshape(s))

        with self.T['Transfer']:
            for transfer in transfer_functions:
                if transfer.func_code.co_argcount == 2:
                    transfer(self.complex, w)
                elif transfer.func_code.co_argcount == 3:
                    transfer(self.comm, self.complex, w)
                else:
                    raise TypeError(
                    "Wrong definition of the transfer function: %s" % transfer.__name__)

    def readout(self, pos):
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
        with self.T['Readout']:
            if pos is not None:
                rt = cic.readout(self.real, pos, mode='ignore', period=self.Nmesh,
                        transform=self.transform)
                return rt
        
    def c2r(self, transfer_functions):
        """ 
        Complex to real transformation.
        
        Parameters
        ----------
        transfer_functions  : list of :py:class:`transfer.TransferFunction`
            A chain of transfer functions to apply to the complex field. 

        """
        self.push()

        self.transfer(*transfer_functions)

        with self.T['C2R']:
            self.backward.execute(self.complex.base, self.real.base)
        if self.verbose:
            realsum = self.comm.allreduce(self.real.sum(dtype='f8'), MPI.SUM)
            if self.comm.rank == 0:
                print 'after c2r, sum of real', realsum
            self.comm.barrier()

        # restore the complex field, for next c2r transform
        self.pop()

