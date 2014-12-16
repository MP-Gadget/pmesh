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
    def __init__(self, Nmesh, comm=None, np=None, verbose=False):
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

    def decompose(self, pos):
        """ create a domain decompose layout for particles at given mesh
            coordinates.

            This layout can be used to migrate particles and images
            to the correct MPI ranks that hosts the PM local mesh

        """
        return self.domain.decompose(pos, smoothing=1.0)

    def r2c(self, pos, mass=1.0):
        """ position is in mesh coordinates, 
            transform the particle field given by pos and mass
            to the overdensity field in fourier space and save
            it in the internal storage.
            particles are first 
        """
        self.real[:] = 0
        pos -= self.partition.local_i_start[None, :]
        cic.paint(pos, self.real, weights=mass, mode='ignore', period=self.Nmesh)
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

    def c2r(self, pos, *transfer_functions):
        """ complex is the fourier space field after applying the transfer 
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
        bak = self.complex.copy()
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
        self.backward.execute(self.complex.base, self.real.base)
        if self.verbose:
            realsum = self.comm.allreduce(self.real.sum(dtype='f8'), MPI.SUM)
            if self.comm.rank == 0:
                print 'after c2r, sum of real', realsum
            self.comm.barrier()
        # restore the complex field, for next c2r transform
        self.complex[:] = bak
        return cic.readout(self.real, pos, mode='ignore', period=self.Nmesh)

class TransferFunction:
    """ these a a function of Window Transfer functions used by PM.
        they take the fourier-space field complex and the dimensionless circular frequency 
        as inputs; complex is modified in-place.

        some functions are factories: they return an actually window with the
        given parameter.

        Working out the dimension of the input is important.
            
        the output of Poisson introduces a dimension to rho_k!
        the output of SuperLanzcos introduces a dimension to rho_k!

    """
    @staticmethod
    def NormalizeDC(comm, complex, w):
        """ removes the DC amplitude. This effectively
            divides by the mean
        """
        ind = []
        value = 0.0
        found = True
        for wi in w:
            if (wi != 0).all():
                found = False
                break
            ind.append((wi == 0).nonzero()[0][0])
        if found:
            ind = tuple(ind)
            value = numpy.abs(complex[ind])
        value = comm.allreduce(value, MPI.SUM)
        complex[:] /= value
    @staticmethod
    def RemoveDC(complex, w):
        ind = []
        for wi in w:
            if (wi != 0).all():
                return
            ind.append((wi == 0).nonzero()[0][0])
        ind = tuple(ind)
        complex[ind] = 0
    @staticmethod
    def Trilinear(complex, w):
        for wi in w:
            # convert to 
            tmp = numpy.sinc(wi / (2 * numpy.pi)) ** 2
            complex[:] /= tmp
    @staticmethod 
    def SuperLanzcos(dir, order=3):
        """ Super Lanzcos kernel of order 3.
            is complex * i * w in a fancier way.

            Notice that for differentiation, one actually wants
            complex * i * k which is
            complex * i * w * Nmesh / BoxSize
        """
        def SuperLanzcosDir(complex, w):
            wi = w[dir] * 1.0
        #    tmp = (1. / 594 * 
        #       (126 * numpy.sin(wi) + 193 * numpy.sin(2 * wi) + 142 * numpy.sin (3 *
        #           wi) - 86 * numpy.sin(4 * wi)))
            tmp = 1 / 6.0 * (8 * numpy.sin (wi) - numpy.sin (2 * wi))
            complex[:] *= tmp * 1j
        return SuperLanzcosDir
    @staticmethod
    def Gaussian(smoothing):
        """ smoothing is in mesh units;
            Notice that this is different from the usual PM split convention.
            (used in Gadget2/3)
            The PM split is cut = sqrt(0.5) * smoothing
        """
        sm2 = smoothing ** 2
        def GaussianS(complex, w):
            w2 = 0
            for wi in w:
                wi2 = wi ** 2
                complex *= numpy.exp(-0.5 * wi2 * sm2)
        return GaussianS
    @staticmethod
    def Constant(C):
        def Constant(complex, w):
            complex *= C
        return Constant
    @staticmethod
    def Inspect(name, *indices):
        """ inspect the complex array at given indices
            mostly for debugging.
        """
        def Inspect(comm, complex, w):
            V = ['%s = %s' %(str(i), str(complex[tuple(i)])) for i in indices]
            print name, ','.join(V)
        return Inspect

    @staticmethod
    def PowerSpectrum(wout, psout):
        """ calculate the power spectrum.
            This shall be done after NormalizeDC and RemoveDC
        """
        def PS(comm, complex, w):
            wedges = numpy.linspace(0, numpy.pi, wout.size + 1, endpoint=True)
            w2edges = wedges ** 2
            w2 = 0.0
            for wi in w:
                w2 = w2 + wi ** 2
            dig = numpy.digitize(w2.flat, w2edges)
            w2 = numpy.bincount(dig, weights=w2.flat, minlength=wout.size + 2)[1: -1]
            w2 = comm.allreduce(w2, MPI.SUM)

            N = numpy.bincount(dig, minlength=wout.size + 2)[1: -1]
            N = comm.allreduce(N, MPI.SUM)

            P = numpy.abs(complex) ** 2
            P = numpy.bincount(dig, weights=P.flat, minlength=wout.size + 2)[1: -1]
            P = comm.allreduce(P, MPI.SUM)

            psout[:] = P / N 
            wout[:] = (w2 / N) ** 0.5
        return PS

    @staticmethod
    def Poisson(complex, w):
        """ 
            Solve Possion equation in k-space: complex /= -w2

            Notes about gravity:

            gravity is 
            
            pot_k = -4pi G delta_k * k **-2 * BoxSize **-3

            where k = w * Nmesh / BoxSize
            hence
            pot_k = -4pi G delta_k * w **-2  * (Nmesh / BoxSize) ** -2 * BoxSize ** -3
                  = -4pi G delta_k * w **-2 * Nmesh ** -2 * BoxSize ** -1

            where this function performs only the -w **-2 part.
        """
        w2 = 0.0
        for wi in w:
            w2 = w2 + wi ** 2
        w2[w2 == 0] = numpy.inf
        w2 *= -1
        complex[:] /= w2

if __name__ == '__main__':

    from gaepsi2.cosmology import WMAP7 as cosmology
    from bigfile import BigFile 
#    from matplotlib import pyplot

    # this will set the units to
    #
    # time: 980 Myear/h
    # distance: 1 Kpc/h
    # speed: 100 km/s
    # mass: 1e10 Msun /h

    DH = 3e5 / 100.
    G = 43007.1
    H0 = 0.1
    Nmesh = 64
    file = BigFile('debug-32/IC')
    header = file.open('header')
    BoxSize = header.attrs['BoxSize'][0]
    a0 = header.attrs['Time'][0]

    Ntot = file.open('1/ID').size
    myslice = slice(
            MPI.COMM_WORLD.rank * Ntot // MPI.COMM_WORLD.size,
            (MPI.COMM_WORLD.rank + 1) * Ntot // MPI.COMM_WORLD.size,
            )
    P = lambda : None
    P.Mass = header.attrs['MassTable'][1]
    P.Pos = file.open('1/Position')[myslice] * (1.0 * Nmesh / BoxSize )
    P.Vel = file.open('1/Velocity')[myslice] * a0 ** 1.5
    P.ID = file.open('1/ID')[myslice] 
    
    NumPart = len(P.Pos)
    print Nmesh, BoxSize, P.Mass
    #NumPart = Nmesh ** 3 // 8 / MPI.COMM_WORLD.size


    pm = ParticleMesh(Nmesh, verbose=False)
    #pos = numpy.random.random(size=(NumPart, 3)) * Nmesh
    #pos = numpy.array(
    #    numpy.indices((Nmesh, Nmesh, Nmesh)), 
    #    dtype='f4').reshape(3, -1).T.copy()
    #MPI.COMM_WORLD.rank * NumPart + numpy.arange(len(pos), dtype='i4')
    #vel = numpy.zeros_like(pos)

    P.Accel = numpy.empty_like(P.Pos)

    def canonical_factors(loga0, loga1, loga2):
        """returns canonical factors for
                kickA, drift, kickB
        """
        N = 129
        g1 = numpy.linspace(loga0, loga1, N, endpoint=True)
        g2 = numpy.linspace(loga1, loga2, N, endpoint=True)
        a1 = numpy.exp(g1)
        a2 = numpy.exp(g2)
        E1 = cosmology.Ea(a1) * H0
        E2 = cosmology.Ea(a2) * H0
        return (
                numpy.trapz(1 / ( a1 * E1), g1),
                numpy.trapz(1 / ( a1 * a1 * E1), g1)
              + numpy.trapz(1 / ( a2 * a2 * E2), g2),
                numpy.trapz(1 / ( a2 * E2), g2),
                )

    dloga = 0.2
    std = None

    loga0 = numpy.log(a0)
    loga = loga0
    vel2 = None
    accel2 = None
    icps = None
    while True:
        # lets get the correct mass distribution with particles on the edge mirrored
        layout = pm.decompose(P.Pos)
        tpos = layout.exchange(P.Pos)
        pm.r2c(tpos, P.Mass)

        # ok. get the smoothed density
        density = pm.c2r(
            tpos, 
            TransferFunction.Constant(BoxSize ** -3),
            TransferFunction.Trilinear,
            TransferFunction.Gaussian(1.25 * 2.0 ** 0.5), 
            TransferFunction.Trilinear,
            )

        wout = numpy.empty(64)
        psout = numpy.empty(64)
        pm.c2r(
            tpos, 
            TransferFunction.Constant(BoxSize ** -3),
            TransferFunction.NormalizeDC,
            TransferFunction.RemoveDC,
            TransferFunction.Trilinear,
    #        TransferFunction.Gaussian(1.25 * 2.0 ** 0.5), 
            # move to Mpc/h units
            TransferFunction.Constant((BoxSize / 1000. / Nmesh) ** 3),
            TransferFunction.PowerSpectrum(wout, psout)
            )

        wout /= (BoxSize / 1000. / Nmesh)
        if icps is None:
            icps = psout.copy()

        if MPI.COMM_WORLD.rank == 0:
            print 'k', wout
            print 'Pk', psout
            print 'power spectrum / IC', psout / icps, \
                (numpy.exp(loga) / numpy.exp(loga0)) ** 2
            #pyplot.plot(wout, psout)
            #pyplot.xscale('log')
            #pyplot.yscale('log')
            #pyplot.draw()
            #pyplot.show()

        density = layout.gather(density, mode='sum')
        Ntot = MPI.COMM_WORLD.allreduce(len(density), MPI.SUM)
        mean = MPI.COMM_WORLD.allreduce(
                numpy.einsum('i->', density, dtype='f8'), MPI.SUM) / Ntot
        std = (MPI.COMM_WORLD.allreduce(numpy.einsum('i,i->', density, density, dtype='f8'), MPI.SUM) /
                Ntot - mean **2)

        dt_kickA, dt_drift, dt_kickB = canonical_factors(
                loga, loga + 0.5 * dloga, loga + dloga)


        for d in range(3):
            tmp = pm.c2r(
                tpos, 
                # to rho_k in comoving units
#                TransferFunction.Inspect('PRE', (0, 0, 1)),
                TransferFunction.Constant(BoxSize ** -3),
                TransferFunction.RemoveDC,
                TransferFunction.Trilinear,
                TransferFunction.Gaussian(1.25 * 2.0 ** 0.5), 
                TransferFunction.Poisson, 
                TransferFunction.Constant(4 * numpy.pi * G),
                TransferFunction.Constant(Nmesh ** -2 * BoxSize ** 2),
                TransferFunction.Trilinear,
#                TransferFunction.Inspect('POT', (0, 0, 1)),
                TransferFunction.SuperLanzcos(d), 
                TransferFunction.Constant(- Nmesh ** 1 * BoxSize ** -1),
#                TransferFunction.Inspect('ACC', (0, 0, 1))
                )
            tmp = layout.gather(tmp, mode='sum')
            # now lets flip the sign of gravity and build a glass
            #tmp *= -1
            P.Accel[:, d] = tmp

        vel2 = MPI.COMM_WORLD.allreduce(numpy.einsum('ij,ij->', P.Vel, P.Vel,
            dtype='f8'), MPI.SUM) ** 0.5
        accel2 = MPI.COMM_WORLD.allreduce(numpy.einsum('ij,ij->', P.Accel, P.Accel,
            dtype='f8'), MPI.SUM) ** 0.5

        if MPI.COMM_WORLD.rank == 0:
            print 'step', \
            'a',  numpy.exp(loga), \
            'mean density', mean, 'std', std, \
            'Ntot', Ntot, 'vel std', vel2, 'accel std', accel2, \
            'dt', dt_kickA, dt_drift, dt_kickB
            print P.Pos[0] / Nmesh * BoxSize, P.Vel[0], P.Accel[0], P.ID[0]

        if loga >= 0.0: break
        
        P.Vel += P.Accel * dt_kickA

        P.Pos += P.Vel * dt_drift * (1. / BoxSize * Nmesh)

        P.Pos %= Nmesh

        P.Vel += P.Accel * dt_kickB

        loga += dloga
