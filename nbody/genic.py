from bigfile import BigFile 
from mpi4py import MPI

from pypm.particlemesh import ParticleMesh
import numpy
from pypm.transfer import TransferFunction

def GridIC(PowerSpectrum, BoxSize, Ngrid, order=3, preshift=False,
        shift=0.5, ZAonly=False, dtype='f8'):
    """ 2LPT IC from PowerSpectrum for particle grid of Ngrid
    
        CPARAM is a Cosmology object. We also need CPARAM.PowerSpectrum object.

        order is the force differentialtion kernel order. 0 or 3.
        This rather long code does 

        (http://arxiv.org/pdf/astro-ph/9711187v1.pdf)

        A few strange things to notice. 
        The real space gaussian field has an amplitude of 1.0. 
        In gaussian field it is 0.707 in amplitude. (grid **3 to adjust that)

        (Also see http://www.design.caltech.edu/erik/Misc/Gaussian.html)
        (And what FFTW really computes)

        After applying the phase each component is further reduced to 0.5.
        (thus FFT back from delta_k with unity power doesn't give us 
        The PowerSpectrum we use is Pk/(2pi)**3. This is the convention used in
        Gadget.

        The sign of terms.  We agree with the paper but shall pull out the - sign in D2
        in Formula D2; 
        The final result agrees with Martin's code(ic_2lpt_big). 
        The final result differ with 2LPTic by -1.

        Factor 3/7 is multiplied to 2LPT field, abs(D2) and abs(D1) shall be
        applied the ZA and 2LPT before shifting the particles and adding the
        velocity.

        Position of initial points. If set to the center of cells the small
        scale power is smoothed. 
        COLA does a global shift after the readout. This matters if one wants to
        evolve the position by 2LPT. We follow COLA, but give an option to do
        the preshift shift.
    """
    # convert to the internal vel units of Gadget a**2 xdot

    D1 = 1.0
    D2 = D1 ** 2 

    pm = ParticleMesh(BoxSize, Ngrid, verbose=False, dtype='f4')

    x0 = pm.partition.local_i_start
    ni = pm.partition.local_ni
    Nlocal = numpy.prod(ni)

    pos = numpy.empty((Nlocal, 3), dtype=dtype)
    ID = numpy.empty(Nlocal, dtype=('i8'))

    view = pos.reshape(list(ni) + [3])
    view[:, :, :, 0] = numpy.arange(ni[0])[:, None, None] + x0[0]
    view[:, :, :, 1] = numpy.arange(ni[1])[None, :, None] + x0[1]
    view[:, :, :, 2] = numpy.arange(ni[2])[None, None, :] + x0[2]

    view *= 1.0 * BoxSize / Ngrid
    if preshift:
        pos += shift * BoxSize / Ngrid

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

    pm.real[:] = RNG.normal(scale=1.0, size=pm.real.shape)
#    realstd = pm.comm.allreduce((pm.real ** 2).sum(), MPI.SUM)
#    if pm.comm.rank == 0:
#        print 'realstd', (realstd / pm.Nmesh ** 3) ** 0.5

    pm.real *= Ngrid ** -1.5

    pm.r2c()
#    realstd = pm.comm.allreduce((pm.complex.real ** 2).sum(), MPI.SUM)
#    if pm.comm.rank == 0:
#        print 'complex std', (realstd / (1. + pm.Nmesh//2 +1) / pm.Nmesh ** 2) ** 0.5

    def Transfer(comm, complex, w):
        w2 = 0
        for wi in w:
            w2 = w2 + wi ** 2
        w2 **= 0.5
        w2 *= 1.0 * Ngrid / BoxSize
        wt = PowerSpectrum.PofK(w2)
        wt *= (2 * numpy.pi) ** 3 * (BoxSize) ** -3 * D1 ** 2
        wt **= 0.5 
        wt[w2 == 0] = 0
        # cut at nyquist
        wt[w2 >= numpy.pi / (BoxSize) * Ngrid] =0 
        complex[:] *= wt

    pm.transfer( [
            TransferFunction.RemoveDC,
            Transfer,
            TransferFunction.Poisson,
            TransferFunction.Constant((1.0 * Ngrid / BoxSize) ** -2),
            ])

    # now we have the 'potential' field in K-space

    # ZA displacements
    P['ZA'] = numpy.empty_like(pos)

    for dir in range(3):
        pm.c2r( [
                TransferFunction.SuperLanzcos(dir, order=order),
                TransferFunction.Constant(-1.0 * Ngrid / BoxSize),
                ])
        tmp = pm.readout(tpos)
        tmp = layout.gather(tmp, mode='sum')
        P['ZA'][:, dir] = tmp

    # additional source term for 2 lpt correction

    # diag terms
    diag = []
    for i, dir in enumerate([(0, 0), (1, 1), (2, 2)]):
        pm.c2r([
                TransferFunction.SuperLanzcos(dir[0], order=order),
                TransferFunction.SuperLanzcos(dir[1], order=order),
                TransferFunction.Constant((1.0 * Ngrid / BoxSize) ** 2),
               ])
        diag.append(pm.real.copy())

    field = diag[0] * diag[1]
    field += diag[1] * diag[2] 
    field += diag[2] * diag[0]
    diag = []

    # off terms
    for i, dir in enumerate([(0, 1), (0, 2), (1, 2)]):
        pm.c2r([
                TransferFunction.SuperLanzcos(dir[0], order=order),
                TransferFunction.SuperLanzcos(dir[1], order=order),
                TransferFunction.Constant((1.0 * Ngrid / BoxSize) ** 2),
               ])
        field -= pm.real ** 2
        
    field *= Ngrid ** -3.0
    pm.real[:] = field
    field = []

    pm.r2c()

    P['2LPT'] = numpy.empty_like(pos)

    tmp = pm.readout(tpos)
    P['digrad'] = layout.gather(tmp, mode='sum')

    for dir in range(3):
        pm.c2r([
                TransferFunction.Poisson,
                TransferFunction.SuperLanzcos(dir, order=0),
                TransferFunction.Constant((1.0 * Ngrid / BoxSize) ** -2),
                TransferFunction.Constant(-1.0 * Ngrid / BoxSize),
                ])
        tmp = pm.readout(tpos)
        tmp = layout.gather(tmp, mode='sum')
        P['2LPT'][:, dir] = tmp

    P['2LPT'] *= 3.0 / 7
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

    stats = dict(
            BoxSize=BoxSize,
            Ngrid=Ngrid,
            stdZA=ZA2 ** 0.5 / BoxSize * Ngrid,
            std2LPT= LPT2 ** 0.5 / BoxSize * Ngrid,
            maxZA= ZAM ** 0.5 / BoxSize * Ngrid,
            max2LPT= LPTM ** 0.5 / BoxSize * Ngrid,
            T=str(pm.T))
    if not preshift:
        P['Position'] += shift * BoxSize / Ngrid
    return P, stats

