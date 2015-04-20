from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser
import numpy
from mpi4py import MPI

from tpm import TPMSnapshotFile, read
from kdcount import cluster
from pypm.domain import GridND
import numpy

parser = ArgumentParser("",
        description=
     "",
        epilog=
     ""
        )

parser.add_argument("filename", 
        help='basename of the input, only runpb format is supported in this script')
parser.add_argument("BoxSize", type=float, 
        help='BoxSize in Mpc/h')
parser.add_argument("LinkingLength", type=float, 
        help='LinkingLength in mean separation (0.2)')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

def main():
    comm = MPI.COMM_WORLD
    np = split_size_2d(comm.size)

    grid = [
        numpy.linspace(0, 1.0, np[0] + 1, endpoint=True),
        numpy.linspace(0, 1.0, np[1] + 1, endpoint=True),
    ]
    domain = GridND(grid)
    logging.info('grid %s' % str(grid) )
    for i, P in enumerate(read(comm, ns.filename, TPMSnapshotFile)):
        pass
    # make sure in one round all particles are in
    assert i == 0

    pos = P['Position'][::1]
    id = P['ID'][::1]

    Ntot = sum(comm.allgather(len(pos)))
    logging.info('Total number of particles %d' % Ntot)
    ll = ns.LinkingLength * Ntot ** -0.3333333
  
    layout = domain.decompose(pos, smoothing=ll)

    tpos = layout.exchange(pos)
    tid = layout.exchange(id)
    logging.info('domain has %d particles' % len(tid))

    idmin = layout.gather(tid, mode=numpy.fmin)
    assert (idmin == id).all()
    print idmin, id
    data = cluster.dataset(tpos, boxsize=1.0)
    fof = cluster.fof(data, linking_length=ll, np=0, verbose=True)
    labels = fof.labels     

    # initialize global labels
    gl = labels + numpy.sum(comm.allgather(fof.N)[0:comm.rank], dtype='intp')

    while True:
        # merge, if a particle belongs to several ranks
        # use the global label of the minimal
        if comm.rank == 0:
            print 'iteration'
        glm = layout.gather(gl, mode=numpy.fmin)
        glm = layout.exchange(glm)

        # on my rank, these particles have been merged
        merged = gl != glm
        # if no rank has merged any, we are done
        # gl is the global label (albeit with some holes)
        if comm.allreduce(merged.sum()) == 0:
            break
        old = gl[merged]
        new = glm[merged]

        # now merge all particles
        permute = numpy.arange(0, gl.max() + 1)
        permute[old] = new
        gl[...] = permute[gl]

    glmg = numpy.concatenate(comm.allgather(glm))
    # linearized glmg
    if comm.rank == 0:
        junk, glmg = numpy.unique(glmg, return_inverse=True)
        N = numpy.bincount(glmg)
        print (N > 32).sum()
    # scatter


    gl = numpy.concatenate(comm.allgather(gl))
    if comm.rank == 0:
        print gl.shape
        print numpy.unique(gl).shape
    #

def split_size_2d(s):
    """ Split `s` into two integers, 
        a and d, such that a * d == s and a <= d

        returns:  a, d
    """
    a = int(s** 0.5) + 1
    d = s
    while a > 1:
        if s % a == 0:
            d = s // a
            break
        a = a - 1 
    return a, d

main()
