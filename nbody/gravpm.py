import sys; import os.path; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy

from bigfile import BigFile 
from mpi4py import MPI

from pypm.transfer import TransferFunction
from pypm.tools import FromRoot
import numba

from genic import GridIC
from qpm import QPM

import logging

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
LOG.addHandler(ch)

def SaveSnapshot(comm, filename, P, blocks=None):
    file = BigFile(filename)
    if blocks is None:
        blocks = P.keys()
    for key in blocks:
        # hack, skip scalar mass
        if numpy.isscalar(P[key]): 
            continue
        file.mpi_create_from_data(comm, '1/%s' % key, P[key])    

def WriteSnapshot(sim, P, aa, prefix):
    pm = sim.pm
    CPARAM = sim.CPARAM
    wout, psout = MeasurePower(pm, P['Position'])
    Ntot = pm.comm.allreduce(len(P['ID']))
    
    D1 = CPARAM.Dplus(aa) / CPARAM.Dplus(1.0)
    D2 = D1 ** 2

    tmp = P['GridPosition'] + D1 * P['ZA']
    tmp %= pm.BoxSize
    P['ZAPosition'] = tmp
    wout1, psout1 = MeasurePower(pm, tmp)

    tmp = P['GridPosition'] + D1 * P['ZA'] + D2 * P['2LPT']
    tmp %= pm.BoxSize
    P['2LPTPosition'] = tmp
    diff = tmp - P['Position']
    diffbar = pm.comm.allreduce(numpy.einsum('ij->j', diff, dtype='f8'), MPI.SUM)
    diffbar /= Ntot
    rms = pm.comm.allreduce(numpy.einsum('ij,ij->', diff, diff, dtype='f8'), MPI.SUM)
    rms /= Ntot
    rms **=0.5

    wout2, psout2 = MeasurePower(pm, tmp)

    if pm.comm.rank == 0:
        LOG.info('Writing snapshot at a=%g z=%g', aa, 1 / aa - 1)
        LOG.info('mean diff bet. 2LPT and QPM: %s', str(diffbar))
        LOG.info('rms diff bet. 2LPT and QPM %g', rms)

        with open(prefix + '/ps-%05.04f.txt' % aa, mode='w') as out:
            numpy.savetxt(out, zip(wout * pm.Nmesh / pm.BoxSize, 
                psout * (6.28 / pm.BoxSize) ** -3,
                psout1 * (6.28 / pm.BoxSize) ** -3,
                psout2 * (6.28 / pm.BoxSize) ** -3, 
                ))

def MeasurePower(pm, pos):
    layout = pm.decompose(pos)
    tpos = layout.exchange(pos)
    pm.r2c(tpos, 1)

    wout = numpy.empty(pm.Nmesh//2)
    psout = numpy.empty(pm.Nmesh//2)
    pm.transfer(
        [
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
        TransferFunction.Trilinear,
#        TransferFunction.Gaussian(1.25 * 2.0 ** 0.5), 
        TransferFunction.PowerSpectrum(wout, psout),
        ]
        )
    return wout, psout

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

def InitialCondition(a0, BoxSize, Ngrid, CPARAM, PowerSpectrum, use_2lpt=False):
    
    D1 = CPARAM.Dplus(a0) / CPARAM.Dplus(1.0)
    D2 = D1 ** 2
    F1 = CPARAM.FOmega(a0)
    F2 = CPARAM.FOmega2(a0)

    P, stats = GridIC(PowerSpectrum, BoxSize, Ngrid, dtype='f4', shift=0.0)

    P['Mass'] = CPARAM.OmegaM * 3 * QPM.H0 * QPM.H0 / ( 8 * numpy.pi * QPM.G) \
            * BoxSize ** 3 / Ngrid ** 3.

    P['GridPosition'] = P['Position'].copy()
    P['Position'] += D1 * P['ZA']
    P['Velocity'] = P['ZA'] * (F1 * D1)
    if use_2lpt:
        P['Position'] += D2 * P['2LPT']
        P['Velocity'] += P['2LPT'] * (2 * F2 * D2)

    P['Position'] %= BoxSize
    P['Velocity'] *= a0 ** 2 * QPM.H0 * CPARAM.Ea(a0)
    P['Accel'] = numpy.zeros_like(P['Position'])

    stats.update(dict(a0=a0, D1=D1, F1=F1, F2=F2, D2=D2, 
            vel_prefac=10 ** 2 * QPM.H0 * CPARAM.Ea(a0)))
    return P, stats

def main():
#    from matplotlib import pyplot
    import cosmology
    CPARAM = cosmology.WMAP9
    import powerspectrum
    from sys import argv
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('--iclevel', dest='iclevel', choices=['za', '2lpt'],
            default='2lpt')
    ap.add_argument('--z0', dest='z0', type=float, default=10.0)
    ap.add_argument('--output', dest='output', default='./')
    ns = ap.parse_args()

    try:
        fh = logging.FileHandler(ns.output + '/log')
        fh.setLevel(logging.INFO)
        LOG.addHandler(fh)
    except IOError:
        LOG.error("cannot open log file at %s", ns.output)
        pass

    # usage gravpm.py 
    a0 = 1 / (float(ns.z0) + 1)
    sim = QPM(
            Nmesh=512,
            BoxSize=256.,
            a0 = a0,
            CPARAM=cosmology.WMAP9)

    aout = 1 / (numpy.array([3.0, 1.0, 0.0]) + 1)

    if sim.comm.rank == 0:
        LOG.info('writing to %s', ns.output)
        LOG.info('IC level: %s', ns.iclevel)
        LOG.info('z0 : %g', ns.z0)

    P, icparam = InitialCondition(a0=a0, 
            BoxSize=sim.BoxSize,
            Ngrid=256,
            CPARAM=cosmology.WMAP9,
            PowerSpectrum=powerspectrum.WMAP9,
            use_2lpt=ns.iclevel == '2lpt')

    if sim.comm.rank == 0:
        LOG.info('%s', str(icparam))

    for event, aa in sim.run(P, aout):
        if event == QPM.PM_STEP_DONE:
            # snapshot and stats
            Ntot = sim.comm.allreduce(len(P['ID']))
            velstd = (sim.comm.allreduce(numpy.einsum('ij,ij->', 
                P['Velocity'],
                P['Velocity'],
                dtype='f8'), MPI.SUM) / Ntot) ** 0.5
            
            if sim.comm.rank == 0:
                LOG.info('Arrived %g ; vel std = %g',
                        aa, velstd)
                LOG.info(str(sim.pm.T))

        elif event == QPM.WRITE_SNAPSHOT:
            WriteSnapshot(sim, P, aa, ns.output)
        elif event == QPM.FINISHED:
            pass
        #SaveSnapshot(pm.comm, 'final-256', P)

if __name__ == '__main__':
    main()
