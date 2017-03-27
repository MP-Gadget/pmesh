"""
    This is a simple Klein Gordon solver.

    It uses the semi-implicit scheme to solve 2D Klein Gordon with pmesh.
    The FFT backend is PFFT.

    The initial condition set up follows Ding 2006

    http://scholarbank.nus.edu.sg/bitstream/handle/10635/15515/main.pdf

    Example 2.3, Ring Solitary Solution.

    The numerical scheme here is described at Equation 2 of Wikibook on Parallel
    Spectrum methods:

    https://en.wikibooks.org/wiki/Parallel_Spectral_Numerical_Methods/The_Klein-Gordon_Equation

    Though that particular chapter may be updated for clarity in the future.

"""

from mpi4py import MPI

from pmesh.pm import ParticleMesh, RealField, ComplexField
import numpy

def kgsolver(steps, u_0, du_0, F=lambda u : -1 * u ** 3, monitor=None):
    """ Solve the Klein-Gordon equation with the non-conserving
        simple semi-implicit scheme.

        discretized version of Klein-Gordon equation used here is

        (1 / dt ** 2 - Nabla ** 2 / 4 + 1 / 4) * u_{n}
        + (- 1 / dt ** 2 - Nabla ** 2 / 4 + 1 / 4) * 2 * u_{n-1}
        + ( 1 / dt ** 2 - Nabla ** 2 + 1 / 4) * u_{n-2} = u_{n-1} ** 3

        Parameters
        ----------
        steps : list of floats:
            A sequence of time steps, the first value, steps[0] is the initial time, where the field
            state is give by u_0.
        u_0 : RealField
            Initial field value
        du_0 : RealField
            Initial time derivative of the field value. This is a second order equation, so we need the
            time derivative. We simply add du_0 * (steps[1] - steps[0]) to jump start the integration.
        F: function
            The nonlinear term of the equation.

    """
    dsteps = numpy.diff(steps)

    u_n_2 = u_0.copy()
    u_n_1 = u_0 + du_0 * dsteps[0]

    if monitor:
        monitor(steps[0], dsteps[0], u_0, du_0)

    for t, dt in zip(steps[1:], dsteps[1:]):
        def transfer_n_2(k, v, dt=dt):
            k2 = sum(ki ** 2 for ki in k)
            factor = (1 / dt ** 2 - 1 / 4.0 * (-k2) + 1 / 4.0)
            return factor * v

        def transfer_n_1(k, v, dt=dt):
            k2 = sum(ki ** 2 for ki in k)
            factor = (-1 / dt ** 2 - 1 / 4.0 * (-k2) + 1 / 4.0)
            return factor * v

        def transfer_n(k, v, dt=dt):
            k2 = sum(ki ** 2 for ki in k)
            # same as transfer_n_2 but divide for the inversion.
            factor = (1 / dt ** 2 - 1 / 4.0 * (-k2) + 1 / 4.0)
            return 1.0 / factor * v

        nonlin = u_n_1.apply(lambda x, v: F(v))

        u_n = (nonlin.r2c(out=Ellipsis)
            - u_n_1.r2c().apply(transfer_n_1, out=Ellipsis)
            - u_n_2.r2c().apply(transfer_n_2, out=Ellipsis))\
            .apply(transfer_n, out=Ellipsis) \
            .c2r(out=Ellipsis)

        if monitor:
            monitor(t, dt, u_n_1, (u_n - u_n_1) / dt)

        # forward the time step
        u_n_2[...] = u_n_1
        u_n_1[...] = u_n

    if monitor:
        monitor(steps[-1], 0, u_n_1, (u_n - u_n_1) / dt)

    return u_n

def main():
    pm = ParticleMesh(BoxSize=32.0, Nmesh=[512, 512], comm=MPI.COMM_WORLD)
    u = pm.create(mode='real')
    def transfer(i, v):
        r = [(ii - 0.5 * ni) * (Li / ni) for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
        r2 = sum(ri ** 2 for ri in r)
        x, y = r
        return 4.0 * numpy.arctan(numpy.exp(3 - r2))
    u = u.apply(transfer, kind='index')

    du = pm.create(mode='real')
    du[...] = 0

    steps = numpy.linspace(0, 16, 201, endpoint=True)

    tmonitor = set([0, 4, 8, 11.5, 15])

    def monitor(t, dt, u, du):
        print("---- timestep %5.3f, step size %5.4f" % (t, dt))
        unorm = u.cnorm()
        print("norm of u is %g." % unorm)

        for tm in tmonitor.copy():
            if (t - tm) * (t + dt - tm) > 0: continue

            preview = u.preview(Nmesh=512, axes=(0, 1))

            if pm.comm.rank == 0:
                print("writing a snapshot")
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                fig = Figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                ax.imshow(preview.T, origin='lower', extent=(0, pm.BoxSize[0], 0, pm.BoxSize[1]))
                canvas = FigureCanvasAgg(fig)
                fig.savefig('klein-gordon-result-%05.3f.png' % t, dpi=128)

            tmonitor.remove(tm)

    kgsolver(steps, u, du, lambda u : numpy.sin(u), monitor=monitor)


if __name__ == '__main__':
    main()

