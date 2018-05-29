from pmesh.lic import lic
from runtests.mpi import MPITest

@MPITest([1, 4])
def test_lic(comm):
    from pmesh.pm import ParticleMesh
    pm = ParticleMesh(Nmesh=[8, 8], comm=comm, BoxSize=8.0)
    vx = pm.create(type='real')
    vy = pm.create(type='real')
    vx = vx.apply(lambda r, v: r[0])
    vy = vy.apply(lambda r, v: 1.0)

    # FIXME: This only tests if the code 'runs', but no correctness
    # is tested.
    r = lic([vx, vy], kernel=lambda s: 1.0, length=2.0, ds=0.1)

