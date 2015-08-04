import numpy
from mpi4py import MPI

def test1():
    comm = MPI.COMM_WORLD
    #0 6 11, 34
  #  rankmapping = [0, 34, 6, 11] + [1] * (comm.size - 4)
    rankmapping = [0, 34, 1, 1] + [1] * (comm.size - 4)
    for batch in [3]:
        rank = rankmapping[comm.rank]
        f1 = numpy.load('/physics2/yfeng1/BWSim/bluetide/test/%d-%03d-layout.npz' %
                (batch, rank))

        sendcounts=f1['sendcounts'][rankmapping]
        recvcounts=f1['recvcounts'][rankmapping]
        allsendcounts = comm.gather(sendcounts)
        allrecvcounts = comm.gather(recvcounts)
        if comm.rank == 0:
            numpy.save('allsendcounts.npy', numpy.array(allsendcounts))
            numpy.save('allrecvcounts.npy', numpy.array(allrecvcounts))
        sendoffsets = sendcounts.copy()
        sendoffsets[0] = 0
        sendoffsets[1:] = sendcounts.cumsum()[:-1]
        recvoffsets = recvcounts.copy()
        recvoffsets[0] = 0
        recvoffsets[1:] = recvcounts.cumsum()[:-1]

        for rr in range(comm.size):
            comm.Barrier()
            if rr != comm.rank: continue
            print comm.rank, '----'
            if (sendcounts != 0).any():
                print 'sendcounts', sendcounts, numpy.nonzero(sendcounts)
            if (recvcounts != 0).any():
                print 'recvcounts', recvcounts, numpy.nonzero(recvcounts)
        data = numpy.ones(100, 'f8') * rank
        newdata = numpy.zeros(100, 'f8') * rank
        comm.Alltoallv((data, sendcounts, sendoffsets, MPI.DOUBLE), 
                            (newdata, recvcounts, recvoffsets, MPI.DOUBLE))
        comm.barrier()
        if comm.rank == 0:
            print 'arrived batch', batch
del test1
