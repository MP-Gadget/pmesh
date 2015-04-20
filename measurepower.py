from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction

from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser
import numpy
from mpi4py import MPI

parser = ArgumentParser("Parallel Power Spectrum Calculator",
        description=
     """Calculating matter power spectrum from RunPB input files. 
        Output is written to stdout, in Mpc/h units. 
        PowerSpectrum is the true one, without (2 pi) ** 3 factor. (differ from Gadget/NGenIC internal)

     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `pypm' which depends on `pfft-python' and mpi4py. 
        The author would like thank Marcel Schmittfull for the explanation on cic, shotnoise, and k==0 plane errors.
     """
        )

parser.add_argument("filename", 
        help='basename of the input, only runpb format is supported in this script')
parser.add_argument("Nmesh", type=int, 
        help='size of calculation mesh, recommend 2 * Ngrid')
parser.add_argument("BoxSize", type=float, 
        help='BoxSize in Mpc/h')
parser.add_argument("--bunchsize", type=int, default=1024*1024*4,
        help='Number of particles to read per rank. A larger number usually means faster IO, but less memory for the FFT mesh')
parser.add_argument("--remove-cic", default='anisotropic', metavar="anisotropic|isotropic|none",
        help='deconvolve cic, anisotropic is the proper way, see http://www.personal.psu.edu/duj13/dissertation/djeong_diss.pdf')
parser.add_argument("--remove-shotnoise", action='store_true', default=False, 
        help='removing the shot noise term')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

def main():
    pm = ParticleMesh(ns.BoxSize, ns.Nmesh)

    Ntot = 0
    for round, P in enumerate(read_tpm(pm.comm, ns.filename, ns.bunchsize)):
        layout = pm.decompose(P['Position'])
        tpos = layout.exchange(P['Position'])
        #print tpos.shape
        pm.paint(tpos)
        npaint = pm.comm.allreduce(len(tpos), op=MPI.SUM) 
        nread = pm.comm.allreduce(len(P['Position']), op=MPI.SUM) 
        if pm.comm.rank == 0:
            logging.info('round %d, npaint %d, nread %d' % (round, npaint, nread))
        Ntot = Ntot + nread

    pm.r2c()
    
    def RemoveShotnoise(complex, w):
        complex[:] -= (1.0 / Ntot) ** 0.5

    def AnisotropicCIC(complex, w):
        for wi in w:
            tmp = (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
            complex[:] /= tmp

    wout = numpy.empty(pm.Nmesh//2)
    psout = numpy.empty(pm.Nmesh//2)

    chain = [
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
    ]

    if ns.remove_cic == 'anisotropic':
        chain.append(AnisotropicCIC)

    if ns.remove_shotnoise:
        chain.append(RemoveShotnoise)

    chain.append(TransferFunction.PowerSpectrum(wout, psout))
        
    # measure the raw power spectrum, nothing is removed.
    pm.push()
    pm.transfer(chain)
    kout = wout * pm.Nmesh / pm.BoxSize
    psout *= (pm.BoxSize) ** 3
    pm.pop()

    if ns.remove_cic == 'isotropic':
        tmp = 1.0 - 0.666666667 * numpy.sin(wout * 0.5) ** 2

    if pm.comm.rank == 0:
        numpy.savetxt(stdout, zip(kout, psout), '%0.7g')

import os.path    
def read_tpm(comm, filename, BunchSize):

    NpartPerFile = []
    if comm.rank == 0:
        i = 0
        while os.path.exists(filename + ".%02d" % i):
            header = numpy.fromfile(filename + ".%02d" % i, dtype='i4', count=7)
            npart = header[2]
            NpartPerFile.append(npart)
            logging.info('found file %s.%02d, npart=%d' % (filename, i, npart))
            i = i + 1
        NpartPerFile = numpy.array(NpartPerFile, dtype='i8')
    else:
        pass
    NpartPerFile = comm.bcast(NpartPerFile)
    NpartCumFile = numpy.concatenate([[0], numpy.cumsum(NpartPerFile)])

    def read_chunk(start, end):
        """this function provides a continuous view of multiple files"""
        pos = []
        id = []
        for i in range(len(NpartPerFile)):
            if end <= NpartCumFile[i]: continue
            if start >= NpartCumFile[i+1]: continue
            # find the intersection in this file
            mystart = max(start - NpartCumFile[i], 0)
            myend = min(end - NpartCumFile[i], NpartPerFile[i])

            with open(filename + ".%02d" %i, 'r') as ff:
                header = numpy.fromfile(ff, dtype='i4', count=7)
                ff.seek(mystart * 12, 1)
                pos.append(numpy.fromfile(ff, count=myend - mystart, dtype=('f4', 3)))
                ff.seek((NpartPerFile[i] - myend )* 12, 1)

                ff.seek(NpartPerFile[i] * 12, 1)
                ff.seek(mystart * 8, 1)
                id.append(numpy.fromfile(ff, count=myend - mystart, dtype=('i8')))
            
        # ensure a good shape even if pos = []
        if len(pos) == 0:
            return (numpy.empty((0, 3), dtype='f4'),
                    numpy.empty((0), dtype='i8'))
        return (numpy.concatenate(pos, axis=0).reshape(-1, 3),
                numpy.concatenate(id, axis=0).reshape(-1))


    Ntot = NpartCumFile[-1]
    mystart = comm.rank * Ntot // comm.size
    myend = (comm.rank + 1) * Ntot // comm.size

    Nchunk = 0
    for i in range(mystart, myend, BunchSize):
        Nchunk += 1
    
    # ensure every rank yields the same number of times
    # for decompose is a collective operation.

    Nchunk = comm.allreduce(Nchunk, op=MPI.MAX)
    for i in range(Nchunk):
        a, b, c = slice(mystart + i * BunchSize, 
                        mystart + (i +1)* BunchSize)\
                    .indices(myend) 
        #print 'round', i, comm.rank, 'reading', a, b
        P = {}
        pos, id = read_chunk(a, b)
        P['Position'] = pos
        P['Position'] *= ns.BoxSize
        #P['ID'] = id
        #print comm.allreduce(P['Position'].max(), op=MPI.MAX)
        #print comm.allreduce(P['Position'].min(), op=MPI.MIN)
        #print P['ID'].max(), P['ID'].min()
        P['Mass'] = 1.0
        yield P
        i = i + BunchSize
        Nchunk = Nchunk - 1


main()
