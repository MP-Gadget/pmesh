import sys; import os.path; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cosmology import WMAP9 as cosmology
from bigfile import BigFile 
from mpi4py import MPI

from pypm.particlemesh import ParticleMesh
import numpy
from pypm.transfer import TransferFunction


if __name__ == '__main__':

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
    Nmesh = 128
    file = BigFile('debug-64/IC')
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
    P.Pos = file.open('1/Position')[myslice] 
    P.Vel = file.open('1/Velocity')[myslice] 
    P.Vel *= a0 ** 1.5
    P.ID = file.open('1/ID')[myslice] 
    
    NumPart = len(P.Pos)
    print Nmesh, BoxSize, P.Mass
    #NumPart = Nmesh ** 3 // 8 / MPI.COMM_WORLD.size


    pm = ParticleMesh(BoxSize, Nmesh, verbose=False)

    P.Accel = numpy.empty_like(P.Pos)

    def canonical_factors(loga0, loga1, loga2):
        """returns canonical factors for
                kickA, drift, kickB
        """
        N = 1025
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


    std = None
    timesteps = list(numpy.linspace(numpy.log(a0), 0.0, 41, endpoint=True))
    vel2 = None
    accel2 = None
    icps = None

    dt_kickA, dt_drift, dt_kickB = None, None, None

    for loga1, loga2 in zip(timesteps[:], timesteps[1:] + [None]):
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
            TransferFunction.PowerSpectrum(wout, psout),
            )

        if icps is None:
            icps = psout.copy()

        if MPI.COMM_WORLD.rank == 0:
            with open('ps-%05.04g.txt' % numpy.exp(loga1), mode='w') as out:
                numpy.savetxt(out, zip(wout * Nmesh / BoxSize, 
                    psout * (6.28 / BoxSize) ** -3))

        density = layout.gather(density, mode='sum')
        Ntot = MPI.COMM_WORLD.allreduce(len(density), MPI.SUM)
        mean = MPI.COMM_WORLD.allreduce(
                numpy.einsum('i->', density, dtype='f8'), MPI.SUM) / Ntot
        std = (MPI.COMM_WORLD.allreduce(numpy.einsum('i,i->', density, density, dtype='f8'), MPI.SUM) /
                Ntot - mean **2)

        if loga2 is None:
            break

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

        if dt_kickB is not None:
            # KickB vel from n+1/2 to n+1
            P.Accel *= dt_kickB
            P.Vel += P.Accel
            P.Accel /= dt_kickB

        # now vel and pos are both at n+1
        vel2 = MPI.COMM_WORLD.allreduce(numpy.einsum('ij,ij->', P.Vel, P.Vel,
            dtype='f8'), MPI.SUM) ** 0.5
        accel2 = MPI.COMM_WORLD.allreduce(numpy.einsum('ij,ij->', P.Accel, P.Accel,
            dtype='f8'), MPI.SUM) ** 0.5

        # next step
        dt_kickA, dt_drift, dt_kickB = canonical_factors(
                loga1, 0.5 * (loga1 + loga2), loga2)

        if MPI.COMM_WORLD.rank == 0:
            print 'step', \
            'a1',  numpy.exp(loga1), \
            'a2',  numpy.exp(loga2), \
            'mean density', mean, 'std', std, \
            'Ntot', Ntot, 'vel std', vel2, 'accel std', accel2, \
            'dt', dt_kickA, dt_drift, dt_kickB
            print P.Pos[0], P.Vel[0], P.Accel[0], P.ID[0]

        # avoid allocating extra memory at the cost of precision.

        # kickA
        # vel n -> n+1/2
        P.Accel *= dt_kickA
        P.Vel += P.Accel
        P.Accel /= dt_kickA

        # drift
        # pos n -> n + 1
        P.Vel *= dt_drift
        P.Pos += P.Vel
        P.Vel /= dt_drift

        # boundary wrap
        P.Pos %= pm.BoxSize

