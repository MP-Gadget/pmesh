import sys; import os.path; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bigfile import BigFile 
from mpi4py import MPI

from pypm.particlemesh import ParticleMesh
import numpy
from pypm.transfer import TransferFunction

from pypm.tools import FromRoot
# this will set the units to
#
# time: 980 Myear/h
# distance: 1 Kpc/h
# speed: 100 km/s
# mass: 1e10 Msun /h

DH = 3e5 / 100.
G = 43007.1
H0 = 0.1

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
    from cosmology import WMAP9 as cosmology
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

def GridIC(BoxSize, Ngrid, a0):
    """ Zel'Dovich IC at a0, for particle grid of Ngrid """
    from cosmology import WMAP9 as cosmology

    pm = ParticleMesh(BoxSize, Ngrid, verbose=False)

    x0 = pm.partition.local_i_start
    ni = pm.partition.local_ni
    Nlocal = numpy.prod(ni)

    pos = numpy.empty(Nlocal, dtype=('f8', 3))
    ID = numpy.empty(Nlocal, dtype=('i8'))

    view = pos.reshape(list(ni) + [3])
    view[:, :, :, 0] = numpy.arange(ni[0])[:, None, None] + x0[0]
    view[:, :, :, 1] = numpy.arange(ni[1])[None, :, None] + x0[1]
    view[:, :, :, 2] = numpy.arange(ni[2])[None, None, :] + x0[2]

    pos += 0.5 * BoxSize / Ngrid
    view *= 1.0 * BoxSize / Ngrid

    # now set up the ranks
    Nlist = numpy.array(pm.comm.allgather(Nlocal), dtype='i8')
    offset = numpy.cumsum(Nlist)
    ID = numpy.arange(Nlocal)
    if pm.comm.rank > 0:
        ID += offset[pm.comm.rank - 1]
    P = dict()
    P['Position'] = pos
    P['ID'] = ID

    layout = pm.decompose(P['Position'])
    tpos = layout.exchange(P['Position'])

    GlobalRNG = numpy.random.RandomState(299995)
    seed = GlobalRNG.randint(999999999, size=pm.comm.size*11)[::11][pm.comm.rank]
    RNG = numpy.random.RandomState(seed)

    pm.real[:] = RNG.normal(scale=Ngrid ** -1.5, size=pm.real.shape)

    pm.r2c()

    D1 = cosmology.Dplus(a0) / cosmology.Dplus(1.0)
    D2 = D1 ** 2 

    def Transfer(complex, w):
        w2 = 0
        for wi in w:
            w2 = w2 + wi ** 2
        w2 **= 0.5
        w2 *= 1.0 * Ngrid / BoxSize
        # Mpc/h
        w2 *= 1000.
        wt = (1e9 * cosmology.PowerSpectrum.PofK(w2)) ** 0.5 

        # this has something to do with K0
        wt *= (2 * numpy.pi / BoxSize) ** 1.5
        wt[w2 == 0] = 0
        # cut at nyquist
        wt[w2 >= numpy.pi / (BoxSize / 1000) * Ngrid] =0 
        complex[:] *= wt

    pm.transfer(
            TransferFunction.RemoveDC,
            TransferFunction.Poisson,
            TransferFunction.Constant((1.0 * Ngrid / BoxSize) ** -2),
            Transfer,
            )
    # now we have the 'potential' field in K-space

    # ZA displacements
    P['ZA'] = numpy.empty_like(pos)

    for dir in range(3):
        tmp = pm.c2r(tpos, 
                TransferFunction.SuperLanzcos(dir),
                TransferFunction.Constant(1.0 * Ngrid / BoxSize),
                )
        tmp = layout.gather(tmp, mode='sum')
        P['ZA'][:, dir] = tmp

    # additional source term for 2 lpt correction

    # diag terms
    diag = []
    for i, dir in enumerate([(0, 0), (1, 1), (2, 2)]):
        pm.c2r(None,
                TransferFunction.SuperLanzcos(dir[0]),
                TransferFunction.SuperLanzcos(dir[1]),
                TransferFunction.Constant(1.0 * Ngrid / BoxSize),
                TransferFunction.Constant(1.0 * Ngrid / BoxSize),
                )
        diag.append(pm.real.copy())

    field = diag[0] * diag[1]
    field += diag[1] * diag[2] 
    field += diag[2] * diag[0]
    diag = []

    # off terms
    for i, dir in enumerate([(0, 1), (0, 2), (1, 2)]):
        pm.c2r(None,
                TransferFunction.SuperLanzcos(dir[0]),
                TransferFunction.SuperLanzcos(dir[1]),
                TransferFunction.Constant(1.0 * Ngrid / BoxSize),
                TransferFunction.Constant(1.0 * Ngrid / BoxSize),
                )
        field -= pm.real ** 2
        
    field *= Ngrid ** -3.0
    pm.real[:] = field
    field = []

    pm.r2c()

    P['2LPT'] = numpy.empty_like(pos)

    tmp = pm.c2r(tpos)
    P['digrad'] = layout.gather(tmp, mode='sum')

    for dir in range(3):
        tmp = pm.c2r(tpos, 
                TransferFunction.Poisson,
                TransferFunction.SuperLanzcos(dir),
                TransferFunction.Constant((1.0 * Ngrid / BoxSize) ** -2),
                TransferFunction.Constant(1.0 * Ngrid / BoxSize),
                )
        tmp = layout.gather(tmp, mode='sum')
        P['2LPT'][:, dir] = tmp

    # convert to the internal vel units of Gadget a**2 xdot
    vel_prefac = a0 ** 2 * H0 * cosmology.Ea(a0)
    F1 = cosmology.FOmega(a0)
    F2 = cosmology.FOmega2(a0)

    # std of displacements
    ZA2 = pm.comm.allreduce(numpy.einsum('ij,ij->', P['ZA'], P['ZA'],
        dtype='f8'), MPI.SUM)
    LPT2 = pm.comm.allreduce(numpy.einsum('ij,ij->', P['2LPT'], P['2LPT'],
        dtype='f8'), MPI.SUM)
    ZAM = pm.comm.allreduce(numpy.max(P['ZA']), MPI.MAX)
    LPTM = pm.comm.allreduce(numpy.max(P['2LPT']), MPI.MAX)

    # norm of the 3-vector!
    ZA2 /= Ngrid ** 3
    LPT2 /= Ngrid ** 3

    P['DISP'] = D1 * P['ZA'] - 3.0 / 7.0 * D2 * P['2LPT']

    DISPM = pm.comm.allreduce(numpy.max(P['DISP']), MPI.MAX)
    if pm.comm.rank == 0:
        print 'BoxSize', BoxSize, 'Ngrid', Ngrid
        print 'a0', a0, \
                'vel_prefac', vel_prefac,\
                'growth',D1 
        print 'F1', F1, 'F2', F2 
            
        print 'ZA std', D1 * ZA2 ** 0.5 / BoxSize * Ngrid
        print '2LPT std', 3.0 / 7.0 * D2 * LPT2 ** 0.5 / BoxSize * Ngrid
        print 'ZA max', D1 * ZAM / BoxSize * Ngrid
        print '2LPT max', 3.0 / 7.0 * D2 * LPTM / BoxSize * Ngrid
        print 'DISP max', DISPM / BoxSize * Ngrid
        print 3.0 / 7.0 * D2 * LPT2 ** 0.5 / (D1 * ZA2 ** 0.5)
        print 'rho_crit=', 3 * H0 * H0 / ( 8 * numpy.pi * G)

    if False:
        P['Position'] += P['DISP']
        P['Position'] %= BoxSize
        P['Velocity'] = vel_prefac* (P['ZA'] * (F1 * D1) - P['2LPT'] * (2 * F2 * D2))
    else:
        P['Position'] += D1 * P['ZA']
        P['Position'] %= BoxSize
        P['Velocity'] = vel_prefac* (P['ZA'] * (F1 * D1))

    P['Mass'] = cosmology.OmegaM * 3 * H0 * H0 / ( 8 * numpy.pi * G) \
            * BoxSize ** 3 / Ngrid ** 3
    return P, BoxSize, a0

def Accel(pm, P, loga):
    # lets get the correct mass distribution with particles on the edge mirrored
    layout = pm.decompose(P['Position'])
    tpos = layout.exchange(P['Position'])
    pm.r2c(tpos, P['Mass'])

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

    if MPI.COMM_WORLD.rank == 0:
        with open('ps-%05.04f.txt' % numpy.exp(loga), mode='w') as out:
            numpy.savetxt(out, zip(wout * pm.Nmesh / pm.BoxSize, 
                psout * (6.28 / pm.BoxSize) ** -3))

    # calculate potential in k-space
    pm.transfer(
            TransferFunction.RemoveDC,
            TransferFunction.Trilinear,
            TransferFunction.Gaussian(1.25 * 4.0 ** 0.5), 
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
    BoxSize = 256000.
    a0 = 1 / 10.
    #P, BoxSize, a0 = ReadIC('debug-64/IC')
    P, BoxSize, a0 = GridIC(BoxSize, 256, a0)
    pm = ParticleMesh(BoxSize, Nmesh, verbose=True)
    SaveSnapshot(pm.comm, 'gridic-256', P)
    MPI.COMM_WORLD.barrier()

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
        
        if pm.comm.rank == 0:
            print 'Step', \
            'a',  numpy.exp(loga1), \
            'vel std', velstd


        if istep == len(timesteps) - 1:
            break
        loga2 = timesteps[istep + 1]

        # prepare next step
        dt_kickA, dt_drift, dt_kickB = canonical_factors(
                loga1, 0.5 * (loga1 + loga2), loga2)

        if pm.comm.rank == 0:
            print 'Next', \
                'a', numpy.exp(loga2), \
                dt_kickA, dt_drift, dt_kickB

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


