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
    
    NumPart = len(P.Pos)
    print Nmesh, BoxSize, P.Mass

    S = strain_tensor(P.Pos, P.Mass, Nmesh, BoxSize, 1000.0)

    S = S.reshape(Ntot, -1)
    with file.create('1/Strain', size=Ntot, dtype=('f4', 9)) as block:
        a, b, junk = myslice.indices(Ntot)
        block.write(a, S)

    block = file.open('1/Strain')
    assert numpy.allclose(S, block[myslice])
