import sys; import os.path; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bigfile import BigFile 
from mpi4py import MPI

from pypm.particlemesh import ParticleMesh
import numpy
from pypm.transfer import TransferFunction
from pypm import cic
from pypm.tools import FromRoot
import cosmology

CPARAM = cosmology.WMAP9
import powerspectrum
PowerSpectrum = powerspectrum.WMAP9

from genic import GridIC
# this will set the units to
#
# time: 98000 Myear/h
# distance: 1 Mpc/h
# speed: 1 km/s
# mass: 1e10 Msun /h

# value of G in potential cancels with particle mass
G = 43007.1
H0 = 100.


try:
    # stop collective mpiimport
    sys.mpiimport.stop()
except AttributeError:
    pass

#@FromRoot(MPI.COMM_WORLD)
# we will only evaluate these numbers from the root
def canonical_factors(loga0, loga1, loga2):
    """returns canonical factors for
            kickA, drift, kickB
    """
    N = 1025
    g1 = numpy.linspace(loga0, loga1, N, endpoint=True)
    g2 = numpy.linspace(loga1, loga2, N, endpoint=True)
    a1 = numpy.exp(g1)
    a2 = numpy.exp(g2)
    E1 = CPARAM.Ea(a1) * H0
    E2 = CPARAM.Ea(a2) * H0
    return (
            numpy.trapz(1 / ( a1 * E1), g1),
            numpy.trapz(1 / ( a1 * a1 * E1), g1)
          + numpy.trapz(1 / ( a2 * a2 * E2), g2),
            numpy.trapz(1 / ( a2 * E2), g2),
            )

def ReadIC(filename):
    # this reads in a MP-Gadget3/GENIC format IC
    # major thing is to scale vel by a0**1.5
    file = BigFile(filename)
    header = file.open('header')
    BoxSize = header.attrs['BoxSize'][0]
    a0 = header.attrs['Time'][0]

    Ntot = file.open('1/ID').size
    myslice = slice(
            MPI.COMM_WORLD.rank * Ntot // MPI.COMM_WORLD.size,
            (MPI.COMM_WORLD.rank + 1) * Ntot // MPI.COMM_WORLD.size,
            )
    P = dict()
    P['Mass'] = header.attrs['MassTable'][1]
    P['Position'] = file.open('1/Position')[myslice] 
    P['Velocity'] = file.open('1/Velocity')[myslice] 
    P['Velocity'] *= a0 ** 1.5
    P['ID'] = file.open('1/ID')[myslice] 
    
    return P, BoxSize, a0

def MeasurePower(pm, pos):
    layout = pm.decompose(pos)
    tpos = layout.exchange(pos)
    pm.r2c(tpos, 1)

    wout = numpy.empty(pm.Nmesh//2)
    psout = numpy.empty(pm.Nmesh//2)
    pm.c2r(
        tpos, 
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
        TransferFunction.Trilinear,
#        TransferFunction.Gaussian(1.25 * 2.0 ** 0.5), 
        TransferFunction.PowerSpectrum(wout, psout),
        )
    return wout, psout

def Accel(pm, P, loga):
    smoothing = 1.0 * pm.Nmesh / BoxSize
    # lets get the correct mass distribution with particles on the edge mirrored
    layout = pm.decompose(P['Position'])
    tpos = layout.exchange(P['Position'])
    pm.r2c(tpos, P['Mass'])

    # calculate potential in k-space
    pm.transfer(
            TransferFunction.RemoveDC,
            TransferFunction.Trilinear,
            TransferFunction.Gaussian(1.25 * smoothing), 
            TransferFunction.Poisson, 
            TransferFunction.Constant(4 * numpy.pi * G),
            TransferFunction.Constant(pm.Nmesh ** -2 * pm.BoxSize ** 2),
    )

    for d in range(3):
        tmp = pm.c2r(
            tpos, 
            TransferFunction.SuperLanzcos(d), 
            # watch out negative for gravity *pulls*!
            TransferFunction.Constant(- pm.Nmesh ** 1 * pm.BoxSize ** -1),
            TransferFunction.Trilinear,
            )
        tmp = layout.gather(tmp, mode='sum')
        P['Accel'][:, d] = tmp

def SaveSnapshot(comm, filename, P, blocks=None):
    file = BigFile(filename)
    if blocks is None:
        blocks = P.keys()
    for key in blocks:
        # hack, skip scalar mass
        if numpy.isscalar(P[key]): 
            continue
        file.mpi_create_from_data(comm, '1/%s' % key, P[key])    

if __name__ == '__main__':

#    from matplotlib import pyplot

    Nmesh = 512
    BoxSize = 256.
    Ngrid = 256
    a0 = 1 / 10.
    #P, BoxSize, a0 = ReadIC('debug-64/IC')
    D1 = CPARAM.Dplus(a0) / CPARAM.Dplus(1.0)
    D2 = D1 ** 2

    P = GridIC(PowerSpectrum, BoxSize, Ngrid, dtype='f4', shift=0.0)

    P['Mass'] = CPARAM.OmegaM * 3 * H0 * H0 / ( 8 * numpy.pi * G) \
            * BoxSize ** 3 / Ngrid ** 3.

    P['ICPosition'] = P['Position'].copy()
    P['Position'] += D1 * P['ZA']
    P['Position'] += D2 * P['2LPT']
    P['Position'] %= BoxSize

    F1 = CPARAM.FOmega(a0)
    F2 = CPARAM.FOmega2(a0)
    P['Velocity'] = a0 ** 2 * H0 * CPARAM.Ea(a0) * \
            (P['ZA'] * (F1 * D1) + P['2LPT'] * (2 * F2 * D2))

    pm = ParticleMesh(BoxSize, Nmesh, verbose=True)
    #SaveSnapshot(pm.comm, 'gridic-256', P)
    MPI.COMM_WORLD.barrier()
    if pm.comm.rank == 0:
        print 'a0', a0, \
                'vel_prefac', a0 ** 2 * H0 * CPARAM.Ea(a0),\
                'growth',D1 
        print 'F1', F1, 'F2', F2 
            

    P['Accel'] = numpy.zeros_like(P['Position'])

    timesteps = list(numpy.linspace(numpy.log(a0), 0.0, 21, endpoint=True))

    for istep in range(len(timesteps)):
        loga1 = timesteps[istep]

        Accel(pm, P, loga1)

        # do the remaining KickB of last step
        if istep > 0:
            # KickB vel from n+1/2 to n+1
            P['Accel'] *= dt_kickB
            P['Velocity'] += P['Accel']
            P['Accel'] /= dt_kickB

        # now vel and pos are both at n+1
        # snapshot and stats
        Ntot = pm.comm.allreduce(len(P['ID']))
        velstd = (pm.comm.allreduce(numpy.einsum('ij,ij->', 
            P['Velocity'],
            P['Velocity'],
            dtype='f8'), MPI.SUM) / Ntot) ** 0.5
        wout, psout = MeasurePower(pm, P['Position'])
        
        D1 = CPARAM.Dplus(numpy.exp(loga1)) / CPARAM.Dplus(1.0)
        D2 = D1 ** 2

        tmp = P['ICPosition'] + D1 * P['ZA']
        tmp %= pm.BoxSize
        P['ZAPosition'] = tmp
        wout1, psout1 = MeasurePower(pm, tmp)

        tmp = P['ICPosition'] + D1 * P['ZA'] + D2 * P['2LPT']
        tmp %= pm.BoxSize
        P['2LPTPosition'] = tmp
        diff = tmp - P['Position']
        diffbar = pm.comm.allreduce(numpy.einsum('ij->j', diff, dtype='f8'), MPI.SUM)
        diffbar /= Ntot
        rms = pm.comm.allreduce(numpy.einsum('ij,ij->', diff, diff, dtype='f8'), MPI.SUM)
        rms /= Ntot
        rms **=0.5

        wout2, psout2 = MeasurePower(pm, tmp)

        if MPI.COMM_WORLD.rank == 0:
            print P['Position'][0], tmp[0]
            with open('ps-%05.04f.txt' % numpy.exp(loga1), mode='w') as out:
                numpy.savetxt(out, zip(wout * pm.Nmesh / pm.BoxSize, 
                    psout * (6.28 / pm.BoxSize) ** -3,
                    psout1 * (6.28 / pm.BoxSize) ** -3,
                    psout2 * (6.28 / pm.BoxSize) ** -3, 
                    ))

        
        if pm.comm.rank == 0:
            print 'Step', \
            'a',  numpy.exp(loga1), \
            'vel std', velstd, \
            'mean diff bet. 2LPT and QPM', diffbar, \
            'rms diff bet. 2LPT and QPM', rms


        if istep == len(timesteps) - 1:
            SaveSnapshot(pm.comm, 'final-256', P)
            break
        loga2 = timesteps[istep + 1]

        # prepare next step
        dt_kickA, dt_drift, dt_kickB = canonical_factors(
                loga1, 0.5 * (loga1 + loga2), loga2)

        if pm.comm.rank == 0:
            print 'Next', \
                'a', numpy.exp(loga2), \
                dt_kickA, dt_drift, dt_kickB
            print str(pm.T)
            print cic.RT
        # kickA
        # vel n -> n+1/2
        P['Accel'] *= dt_kickA
        P['Velocity'] += P['Accel']
        P['Accel'] /= dt_kickA

        # drift
        # pos n -> n + 1
        P['Velocity'] *= dt_drift
        P['Position'] += P['Velocity']
        P['Velocity'] /= dt_drift

        # boundary wrap
        P['Position'] %= pm.BoxSize


