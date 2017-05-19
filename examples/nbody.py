from mpi4py import MPI
import numpy

from argparse import ArgumentParser

from nbodykit.cosmology import Planck15
from nbodykit.cosmology import EHPower
from nbodykit.cosmology.perturbation import PerturbationGrowth
from scipy.integrate import quad
PowerSpectrum = EHPower(Planck15, redshift=0.0)
pt = PerturbationGrowth(Planck15.clone(Tcmb0=0))

class FastPM:
    def K(ai, af, ar):
        return 1 / (ar ** 2 * pt.E(ar)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ar)
    def D(ai, af, ar):
        return 1 / (ar ** 3 * pt.E(ar)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ar)

class FastPM1:
    def K(ai, af, ar):
        def func(a):
            return 1.0 / (a * a * pt.E(a))
        return quad(func, ai, af)[0]
    def D(ai, af, ar):
        return 1 / (ar ** 3 * pt.E(ar)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ar)

class FastPM2:
    def K(ai, af, ar):
        return 1 / (ar ** 2 * pt.E(ar)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ar)
    def D(ai, af, ar):
        def func(a):
            return 1.0 / (a * a * a * pt.E(a))
        return quad(func, ai, af)[0]

class Quinn:
    def K(ai, af, ar):
        def func(a):
            return 1.0 / (a * a * pt.E(a))
        return quad(func, ai, af)[0]
    def D(ai, af, ar):
        def func(a):
            return 1.0 / (a * a * a * pt.E(a))
        return quad(func, ai, af)[0]

class TVE:
    """ split H = T + (E + V); drift has no explicit time dependency """
    def K(ai, af, ar):
        def func(a):
            return 1.0 / (a * a * pt.E(a))
        return quad(func, ai, af)[0]
    def D(ai, af, ar):
        def func(a):
            return 1.0 / (a * pt.E(a))
        return ar ** -2 * quad(func, ai, af)[0]

class VTE:
    """ split H = (T + E) + V; kick has no explicit time dependency """
    def K(ai, af, ar):
        def func(a):
            return 1.0 / (a * pt.E(a))
        return ar ** -1 * quad(func, ai, af)[0]
    def D(ai, af, ar):
        def func(a):
            return 1.0 / (a * a * a * pt.E(a))
        return quad(func, ai, af)[0]


class State:
    def __init__(self, Q, S, V):
        self.Q = Q
        self.S = S
        self.V = V

def symp2(pm, state, time_steps, factors):
    K = factors.K
    D = factors.D
    Q = state.Q
    V = state.V
    S = state.S

    F = force(pm, Q, S)
    E = 0
    for ai, af in zip(time_steps[:-1], time_steps[1:]):
        ac = (ai * af) ** 0.5
        V[...] += F * K(ai, ac, ai)
        S[...] += V * D(ai, af, ac)
        F[...] = force(pm, Q, S)
        V[...] += F * K(ac, af, af)

        print(af)
        #E = energy(pm, Q, S, V, af)
        #print('E = ', E, af)

def symp3(pm, state, time_steps, factors):
    K = factors.K
    D = factors.D
    Q = state.Q
    V = state.V
    S = state.S

    F = force(pm, Q, S)
    for ai, af in zip(time_steps[:-1], time_steps[1:]):
        Dloga = numpy.log(af) - numpy.log(ai)
        ac1 = af
        ac2 = ac1 * numpy.exp(- 2.0 / 3.0 * Dloga)
        ac3 = af

        ad1 = ai * numpy.exp(- 1. / 24 * Dloga)
        ad2 = ad1 * numpy.exp(3. / 4 * Dloga)
        ad3 = af

        S[...] += V * D(ai, ac1, ai)
        F[...] = force(pm, Q, S)
        V[...] += F * K(ai, ad1, af)
        S[...] += V * D(af, ac2, ad1)
        F[...] = force(pm, Q, S)
        V[...] += F * K(ad1, ad2, ac2)
        S[...] += V * D(ac2, ac3, ad2)
        F[...] = force(pm, Q, S)
        V[...] += F * K(ad2, ad3, ac3)

        print(af)
        #E = energy(pm, Q, S, V, af)
        #print('E = ', E, af)


def symp1(pm, state, time_steps, factors):
    K = factors.K
    D = factors.D
    Q = state.Q
    V = state.V
    S = state.S

    F = force(pm, Q, S)
    for ai, af in zip(time_steps[:-1], time_steps[1:]):
        F = force(pm, Q, S)
        V[...] += F * K(ai, af, ai)
        S[...] += V * D(ai, af, af)
        F[...] = force(pm, Q, S)
        #E = energy(pm, Q, S, V, af)
        #print('E = ', E, af)
        print(af)

def dx1_transfer(direction):
    def filter(k, v):
        k2 = sum(ki ** 2 for ki in k)
        k2[k2 == 0] = 1.0
        kfinite = k[direction]
        return 1j * kfinite / k2 * v
    return filter

def force_transfer(direction):
    def filter(k, v):
        k2 = sum(ki ** 2 for ki in k)
        k2[k2 == 0] = 1.0
        C = (v.BoxSize / v.Nmesh)[direction]
        w = k[direction] * C
        kfinite = 1.0 / C * 1 / 6.0 * (8 * numpy.sin (w) - numpy.sin (2 * w));
        return 1j * kfinite / k2 * v
    return filter

def pot_transfer(k, v):
    k2 = sum(ki ** 2 for ki in k)
    k2[k2 == 0] = 1.0
    return -1. / k2 * v

def lowpass_transfer(r):
    def filter(k, v):
        k2 = sum(ki ** 2 for ki in k)
        return numpy.exp(-0.5 * k2 * r**2) * v
    return filter

from pmesh.pm import ParticleMesh


def main(ns):
    comm = MPI.COMM_WORLD

    result = simulate(comm, ns)
 
    pm = ParticleMesh(BoxSize=ns.BoxSize, Nmesh=[ns.Nmesh, ns.Nmesh, ns.Nmesh], dtype='f8', comm=comm)
    report = analyze(pm, result)

    if comm.rank == 0:
        write_report(ns.output, report)

class Result(object): pass

def force(pm, Q, S):
    rho1 = pm.create('real')
    X = S + Q
    layout = pm.decompose(X, smoothing=1.0 * pm.method.support)
    rho1.paint(X, layout=layout, hold=False)

    N = pm.comm.allreduce(len(X))
    fac = 1.0 * pm.Nmesh.prod() / N
    rho1[...] *= fac
    rhok1 = rho1.r2c()

    rhok = rhok1
    #rhok.apply(CompensateTSCAliasing, kind='circular', out=Ellipsis)

    #print(fac, rhok.cgetitem([0, 0, 0]), rhok.cgetitem([1, 1, 1]))
    F = numpy.empty_like(Q)
    for d in range(pm.ndim):
        F[..., d] = rhok.apply(force_transfer(d)) \
                  .c2r().readout(X, layout=layout)
    return 1.5 * pt.Om0 * F

def energy(pm, Q, S, V, a):
    rho1 = pm.create('real')
    X = S + Q
    layout = pm.decompose(X, smoothing=1.0 * pm.method.support)
    rho1.paint(X, layout=layout, hold=False)

    N = pm.comm.allreduce(len(X))
    fac = 1.0 * pm.Nmesh.prod() / N
    rho1[...] *= fac
    rhok1 = rho1.r2c()
    phi = rhok1.apply(pot_transfer) \
              .apply(lowpass_transfer(pm.BoxSize[0] / pm.Nmesh[0] * 4)) \
              .c2r().readout(X, layout=layout)

    U = 1.5 * pt.Om0 * pm.comm.allreduce(phi.sum() / a)

    T = 0
    for d in range(pm.ndim):
        rho1.paint(Q, mass=V[:, d], hold=False)
        V1 = rho1.r2c() \
              .apply(lowpass_transfer(pm.BoxSize[0] / pm.Nmesh[0] * 4)) \
              .c2r().readout(Q)
        T = T + pm.comm.allreduce((V1 ** 2).sum() / (2 * a**2))
    return T + U

def simulate(comm, ns):

    pm = ParticleMesh(BoxSize=ns.BoxSize, Nmesh=[ns.Nmesh, ns.Nmesh, ns.Nmesh], dtype='f8', comm=comm)
    gaussian = pm.generate_whitenoise(ns.seed, unitary=True)
    time_steps = numpy.linspace(ns.ainit, ns.afinal, ns.steps, endpoint=True)

    Q = pm.generate_uniform_particle_grid(shift=0)
    print(Q.min(axis=0), Q.max(axis=0))
    def convolve(k, v):
        kmag = sum(ki**2 for ki in k) ** 0.5
        ampl = (PowerSpectrum(kmag) / v.BoxSize.prod()) ** 0.5
        return v * ampl
        
    dlinear = gaussian.apply(convolve)


    DX1 = numpy.zeros_like(Q)
    layout = pm.decompose(Q)
    # Fill it in one dimension at a time.
    for d in range(pm.ndim):
        DX1[..., d] = dlinear \
                      .apply(dx1_transfer(d)) \
                      .c2r().readout(Q, layout=layout)


    a0 = time_steps[0]

    # 1-LPT Displacement and Veloicty; scaled back from z=0 to the first time step.
    S = DX1 * pt.D1(a=a0)
    V = S * a0 ** 2 * pt.f1(a0) * pt.E(a0)
    state = State(Q, S, V)

    fpm = ParticleMesh(BoxSize=pm.BoxSize, Nmesh=pm.Nmesh * ns.boost, method='tsc', dtype='f8')

    ns.scheme(fpm, state, time_steps, ns.factors)
        
    r = Result()
    r.Q = Q    
    r.DX1 = DX1
    r.S = S
    r.V = V
    r.dlinear = dlinear
    
    return r

def analyze(pm, r):
    from nbodykit.algorithms.fftpower import FFTPower
    from nbodykit.source import ArrayCatalog
    from nbodykit.source import MemoryMesh

    DataPM = numpy.empty(len(r.Q), dtype=[('Position', ('f8', 3))])
    DataPM['Position'][:] = r.Q + r.S

    Data1LPT = DataPM.copy()
    Data1LPT['Position'][:] = r.Q + r.DX1

    DataPM = ArrayCatalog(DataPM, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
    Data1LPT = ArrayCatalog(Data1LPT, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
    DataLinear = MemoryMesh(r.dlinear)
    r = Result()
    r.Ppm = FFTPower(DataPM, mode='1d')
    r.P1lpt = FFTPower(Data1LPT, mode='1d')
    r.Pl = FFTPower(DataLinear, mode='1d')
    return r

def write_report(reportname, r):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(r.Ppm.power['k'], r.Ppm.power['power'] / r.Pl.power['power'] - 1, label='Multistep')
    ax.plot(r.P1lpt.power['k'], r.P1lpt.power['power'] / r.Pl.power['power'] - 1, label='1-LPT')
    ax.set_xscale('log')
    ax.axhline(0.0, color='k', ls='--')
    ax.set_ylim(-0.03, 0.03)
    ax.set_xlim(0.003, 0.04)
    ax.grid()
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel(r'P(k) / <P_l(k)>')
    ax.set_title(r"Comparing Linear theory and 1-LPT")
    ax.legend()

#    numpy.savez(reportname.replace('.png', '.npz', r.__dict__)
    canvas = FigureCanvasAgg(fig)
    fig.savefig(reportname)

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--Nmesh", type=int, default=64)
    ap.add_argument("--BoxSize", type=float, default=200.)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--ainit", type=float, default=0.1)
    ap.add_argument("--afinal", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=120577)
    ap.add_argument("--boost", type=int, default=2)
    ap.add_argument("--scheme", choices=[symp2, symp1, symp3], default=symp2, type=lambda n: globals()[n])
    ap.add_argument("--factors", choices=[FastPM, FastPM1, FastPM2, Quinn, VTE, TVE], default=FastPM,
            type=lambda n : globals()[n])
    ap.add_argument("output", type=str)


    ns = ap.parse_args()

    main(ns)
