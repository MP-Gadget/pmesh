from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction

from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser
import numpy
from mpi4py import MPI

parser = ArgumentParser()

parser.add_argument("filename")
parser.add_argument("Nmesh", type=int)
parser.add_argument("BoxSize", type=float)

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

def main():
    pm = ParticleMesh(ns.BoxSize, ns.Nmesh)

    Ntot = 0
    for round, P in enumerate(read_tpm(pm.comm, ns.filename)):
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
    
    def RemoveShotNoise(complex, w):
        complex[:] -= 1 / Ntot

    wout = numpy.empty(pm.Nmesh//2)
    psout0 = numpy.empty(pm.Nmesh//2)
    psout1 = numpy.empty(pm.Nmesh//2)
    psout2 = numpy.empty(pm.Nmesh//2)

    pm.push()
    pm.transfer( [
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
        RemoveShotNoise,
        TransferFunction.PowerSpectrum(wout, psout0),
    ])
    wout *= pm.Nmesh / pm.BoxSize
    psout0 *= (numpy.pi / pm.BoxSize) ** -3
    pm.pop()

    pm.push()
    pm.transfer( [
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
        RemoveShotNoise,
        TransferFunction.Trilinear,
        TransferFunction.PowerSpectrum(wout, psout1),
    ])
    wout *= pm.Nmesh / pm.BoxSize
    psout1 *= (numpy.pi / pm.BoxSize) ** -3
    pm.pop()

    pm.push()
    pm.transfer( [
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
        RemoveShotNoise,
        TransferFunction.PowerSpectrum(wout, psout2),
    ])
    pm.pop()

    tmp = 1.0 - numpy.sin(wout * 0.5) ** 2

    wout *= pm.Nmesh / pm.BoxSize
    psout2 *= (numpy.pi / pm.BoxSize) ** -3
    psout2 /= tmp

#    psout1 -= 1. / Ntot
#    psout2 -= 1. / Ntot
    if pm.comm.rank == 0:
        numpy.savetxt(stdout, zip(wout, psout0, psout1, psout2))

import os.path    
def read_tpm(comm, filename):
    # read this many per chunk
    BunchSize = 1024 * 1024

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
