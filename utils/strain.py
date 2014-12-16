import sys; import os.path; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gaepsi2.cosmology import WMAP7 as cosmology
from bigfile import BigFile 
from mpi4py import MPI

from pypm.particlemesh import ParticleMesh
import numpy
from pypm.transfer import TransferFunction

def strain_tensor(Pos, Mass, Nmesh, BoxSize, smoothing):
    """ Pos and smoothing is given in the same unit as BoxSize """
    Ndim = Pos.shape[1]

    assert Ndim == 3

    # first convert to Nmesh units:

    smoothing = smoothing * (1.0 * Nmesh / BoxSize)

    Pos = Pos * (1.0 * Nmesh / BoxSize)

    pm = ParticleMesh(Nmesh, verbose=False)

    layout = pm.decompose(Pos)
    tpos = layout.exchange(Pos)

    if numpy.isscalar(P.Mass):
        tmass = P.Mass
    else:
        tmass = layout.exchange(P.Mass)

    pm.r2c(tpos, tmass)

    S = numpy.empty((len(Pos), 
                Ndim, Ndim), dtype='f8')

    for i, j in numpy.ndindex(Ndim, Ndim):
        if i > j: continue
        tmp = pm.c2r(
            tpos, 
            TransferFunction.Constant(BoxSize ** -3),
            TransferFunction.RemoveDC,
            TransferFunction.Trilinear,
            TransferFunction.Gaussian(smoothing), 
            TransferFunction.Poisson, 
            TransferFunction.Constant(4 * numpy.pi * G),
            TransferFunction.Constant(Nmesh ** -2 * BoxSize ** 2),
            TransferFunction.Trilinear,
            TransferFunction.SuperLanzcos(i), 
            TransferFunction.SuperLanzcos(j), 
            TransferFunction.Constant(Nmesh ** 1 * BoxSize ** -1),
            TransferFunction.Constant(Nmesh ** 1 * BoxSize ** -1),
            )

        tmp = layout.gather(tmp, mode='sum')
        # symmetric!
        S[..., i, j] = tmp
        S[..., j, i] = tmp
    return S

def overdensity(Pos, Mass, Nmesh, BoxSize, smoothing):
    """ Pos and smoothing is given in the same unit as BoxSize """
    Ndim = Pos.shape[1]

    assert Ndim == 3

    # first convert to Nmesh units:

    smoothing = smoothing * (1.0 * Nmesh / BoxSize)

    Pos = Pos * (1.0 * Nmesh / BoxSize)

    pm = ParticleMesh(Nmesh, verbose=False)

    layout = pm.decompose(Pos)
    tpos = layout.exchange(Pos)

    if numpy.isscalar(P.Mass):
        tmass = P.Mass
    else:
        tmass = layout.exchange(P.Mass)

    pm.r2c(tpos, tmass)

    D = numpy.empty(len(Pos), dtype='f8')

    tmp = pm.c2r(
        tpos, 
        TransferFunction.Constant(BoxSize ** -3),
        TransferFunction.Inspect('K0', (0, 0, 0)),
#        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
        TransferFunction.Trilinear,
        TransferFunction.Gaussian(smoothing), 
        TransferFunction.Trilinear,
        )
    D[:] = layout.gather(tmp, mode='sum')
    return D

if __name__ == '__main__':
    from sys import argv

    scales = [int(a) for a in argv[1:]]

    def mpicreate(file, blkname, size, dtype, comm):
        if comm.rank == 0:
            file.create(blkname, size=size, dtype=dtype)
        comm.barrier()
        return file.open(blkname)

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
    P.Pos = file.open('1/Position')[myslice] 
    
    NumPart = len(P.Pos)
    print Nmesh, BoxSize, P.Mass

    for scale in scales:
        S = strain_tensor(P.Pos, P.Mass, Nmesh, BoxSize, 1.0 * scale)
        S = S.reshape(NumPart, -1)
        with mpicreate(file, '1/Strain-%d' % scale, size=Ntot, dtype=('f4', 9), comm=MPI.COMM_WORLD) as block:
            a, b, junk = myslice.indices(Ntot)
            block.write(a, S)
        D = overdensity(P.Pos, P.Mass, Nmesh, BoxSize, 1.0 * scale)
        with mpicreate(file, '1/OverDensity-%d' % scale, size=Ntot, dtype=('f4'), comm=MPI.COMM_WORLD) as block:
            a, b, junk = myslice.indices(Ntot)
            block.write(a, D)
