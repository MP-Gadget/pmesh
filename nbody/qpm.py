import sys; import os.path; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bigfile import BigFile 
from mpi4py import MPI

from pypm.particlemesh import ParticleMesh
import numpy
from pypm.transfer import TransferFunction
from pypm.tools import FromRoot
import numba


class QPM(object):
    # value of G in potential cancels with particle mass

    # this will set the units to
    #
    # time: 98000 Myear/h
    # distance: 1 Mpc/h
    # speed: 1 km/s
    # mass: 1e10 Msun /h

    G = 43007.1
    H0 = 100.
    PM_STEP_DONE = 1
    WRITE_SNAPSHOT = 2
    FINISHED = 3
    def __init__(self, CPARAM, 
            BoxSize, Nmesh, a0, comm=None):
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = None

        self.a0 = a0
        self.CPARAM = CPARAM
        self.Nmesh = Nmesh
        self.BoxSize = BoxSize

    def Kick(self, P, loga0, loga1):
#        if self.comm.rank == 0:
#            print 'kicking from', numpy.exp(loga0), 'to', numpy.exp(loga1)
        CPARAM = self.CPARAM
        N = 1025
        g1 = numpy.linspace(loga0, loga1, N, endpoint=True)
        a1 = numpy.exp(g1)
        E1 = CPARAM.Ea(a1) * self.H0
        dt_kick = numpy.trapz(1 / ( a1 * E1), g1)
        self.RealKick(P['Velocity'], P['Accel'], dt_kick)

    @staticmethod
    @numba.jit(nopython=True)
    def RealKick(vel, acc, dt_kick):
        for i in range(vel.shape[0]):
            for d in range(vel.shape[1]):
                vel[i, d] += acc[i, d] * dt_kick

    def Drift(self, P, loga0, loga1):
#        if self.comm.rank == 0:
#            print 'drifting from', numpy.exp(loga0), 'to', numpy.exp(loga1)
        CPARAM = self.CPARAM
        BoxSize = self.BoxSize

        N = 1025
        g1 = numpy.linspace(loga0, loga1, N, endpoint=True)
        a1 = numpy.exp(g1)
        E1 = CPARAM.Ea(a1) * self.H0
        dt_drift = numpy.trapz(1 / (a1 * a1 * E1), g1)
        self.RealDrift(P['Position'], P['Velocity'], BoxSize, dt_drift)

    @staticmethod
    @numba.jit(nopython=True)
    def RealDrift(pos, vel, BoxSize, dt_drift):
        for i in range(pos.shape[0]):
            for d in range(pos.shape[1]):
                pos[i, d] += vel[i, d] * dt_drift
                while pos[i, d] < 0:
                    pos[i, d] += BoxSize
                while pos[i, d] >= BoxSize:
                    pos[i, d] -= BoxSize

    @staticmethod
    def Accel(pm, P):
        smoothing = 1.0 * pm.Nmesh / pm.BoxSize
        # lets get the correct mass distribution with particles on the edge mirrored
        layout = pm.decompose(P['Position'])
        tpos = layout.exchange(P['Position'])
        pm.r2c(tpos, P['Mass'])

        # calculate potential in k-space
        pm.transfer(
                TransferFunction.RemoveDC,
                TransferFunction.Trilinear,
                TransferFunction.Gaussian(1.25 * smoothing), 
                TransferFunction.Poisson, 
                TransferFunction.Constant(4 * numpy.pi * QPM.G),
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

    def run(self, P, aout=[]):
        logaout = numpy.log(aout)
        logaout.sort()

        pm = ParticleMesh(self.BoxSize, self.Nmesh, verbose=False)
        self.pm = pm
        #SaveSnapshot(pm.comm, 'gridic-256', P)

        dloga = 0.1
        timesteps = list(numpy.arange(numpy.log(self.a0), 0.0, dloga))
        if timesteps[-1] < 0.0:
            timesteps.append(timesteps[-1] + dloga)

        for istep in range(len(timesteps)):

            # force at x(n+1)
            self.Accel(pm, P)

            # do the remaining KickB of last step
            if istep > 0:
                # KickB vel from n+1/2 to n+1
                self.Kick(P, 0.5 * (loga1 + loga2), loga2)

            loga1 = timesteps[istep]

            if istep == len(timesteps) - 1:
                # no more steps
                break
            if loga1 > logaout.max():
                # no more output
                break

            # now vel and pos are both at n+1, notify the caller! 
            yield self.PM_STEP_DONE, numpy.exp(loga1)

            loga2 = timesteps[istep + 1]

            # kickA
            # vel n -> n+1/2
            self.Kick(P, loga1, 0.5 * (loga1 + loga2))

            # drift
            # pos n -> n + 1

            # detect snapshot times
            left = logaout.searchsorted(loga1, side='left')
            right = logaout.searchsorted(loga2, side='right')

            if left != right:
                self.Drift(P, loga1, logaout[left])
                yield self.WRITE_SNAPSHOT, numpy.exp(logaout[left])
                for i in range(left + 1, right):
                    self.Drift(P, logaout[i-1], logaout[i])
                    yield self.WRITE_SNAPSHOT, numpy.exp(logaout[i])
                self.Drift(P, logaout[right - 1], loga2)
            else:
                self.Drift(P, loga1, loga2)

        yield self.FINISHED, numpy.exp(loga1)
