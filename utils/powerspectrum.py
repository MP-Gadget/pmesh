import sys; import os.path; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gaepsi2.cosmology import WMAP7 as cosmology
from bigfile import BigFile 
from mpi4py import MPI

from pypm.particlemesh import ParticleMesh
from sys import stdout
from pypm import cic
import numpy
from pypm.transfer import TransferFunction

if __name__ == '__main__':
    from sys import argv

    # this will set the units to
    #
    # time: 980 Myear/h
    # distance: 1 Kpc/h
    # speed: 100 km/s
    # mass: 1e10 Msun /h

    DH = 3e5 / 100.
    G = 43007.1
    H0 = 0.1
    Nmesh = 32
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
    
    NumPart = len(P.Pos)
    print '#', Nmesh, BoxSize, P.Mass

    P.Pos *= (1.0 * Nmesh / BoxSize)

    pm = ParticleMesh(Nmesh, verbose=False)
    layout = pm.decompose(P.Pos)
    tpos = layout.exchange(P.Pos)
    if numpy.isscalar(P.Mass):
        tmass = P.Mass
    else:
        tmass = layout.exchange(P.Mass)

    pm.r2c(tpos, tmass)

    psout = numpy.empty(Nmesh)
    wout = numpy.empty(Nmesh)

    pm.transfer( 
            TransferFunction.Trilinear,
            TransferFunction.NormalizeDC,
            TransferFunction.RemoveDC,
            TransferFunction.PowerSpectrum(wout, psout),
            )

    numpy.savetxt(stdout, zip(wout * Nmesh /  BoxSize, psout * (6.28 / BoxSize)
        ** -3))
