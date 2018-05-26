from runtests.mpi import MPITest
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing.decorators import skipif

from pmesh.pm import ParticleMesh, RealField, ComplexField
from pmesh import window
import numpy

@MPITest(commsize=(1,))
def test_asarray(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)
    a = numpy.asarray(real)
    assert a is real.value

    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f4')
    real = RealField(pm)
    a = numpy.asarray(real)
    assert a is real.value

    real = RealField(pm)
    a = numpy.array(real, copy=False)
    assert a is real.value

@MPITest(commsize=(1, 4))
def test_shape_real(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)
    assert (tuple(real.cshape) == (8, 8))

@MPITest(commsize=(1, 4))
def test_shape_complex(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    comp = ComplexField(pm)
    assert (tuple(comp.cshape) == (8, 5))

@MPITest(commsize=(1,))
def test_negnyquist(comm):
    # the nyquist mode wave number in the hermitian complex field must be negative.
    # nbodykit depends on this behavior.
    # see https://github.com/bccp/nbodykit/pull/459
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    c = pm.create(mode='complex')
    assert (c.x[-1][0][-1] < 0).all()
    assert (c.x[-1][0][:-1] >= 0).all()

@MPITest(commsize=(1,))
@skipif(True, "1d is not supported")
def test_1d(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8], comm=comm, dtype='f8')
    real = pm.generate_whitenoise(seed=123, mode='real')
    complex = pm.generate_whitenoise(seed=123, mode='complex')
    assert_array_equal(real, complex.c2r())

@MPITest(commsize=(4,))
def test_2d_2d(comm):
    import pfft
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], np=pfft.split_size_2d(comm.size), comm=comm, dtype='f8')
    pm2 = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8, 8], np=pfft.split_size_2d(comm.size), comm=comm, dtype='f8')
    assert pm._use_padded == False
    real = pm.generate_whitenoise(seed=123, mode='real')
    complex = pm.generate_whitenoise(seed=123, mode='complex')
    assert_array_equal(real, complex.c2r())

    real2 = pm2.generate_whitenoise(seed=123, mode='real')

    assert real2.shape[:2] == real.shape

@MPITest(commsize=(1,4))
def test_fft(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f4')
    numpy.random.seed(1234)

    real = pm.create(mode='real', value=0)
    raw = real.base.view_raw()
    real[...] = 2
    real[::2, ::2] = -2
    real3 = real.copy()
    complex = real.r2c()
    assert_almost_equal(numpy.asarray(real), numpy.asarray(real3), decimal=7)

    real2 = complex.c2r()
    assert_almost_equal(numpy.asarray(real), numpy.asarray(real2), decimal=7)

@MPITest(commsize=(1,4))
def test_whitenoise_untransposed(comm):
    pm_ut = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f4', transposed=False)
    pm_t = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f4', transposed=True)

    f1 = pm_ut.generate_whitenoise(seed=3333, mode='complex')
    f2 = pm_t.generate_whitenoise(seed=3333, mode='complex')

    f1r = numpy.concatenate(comm.allgather(numpy.array(f1.ravel())))
    f2r = numpy.concatenate(comm.allgather(numpy.array(f2.ravel())))

    assert_array_equal(f1r, f2r)

    # this should have asserted r2c transforms as well.
    r1 = pm_ut.generate_whitenoise(seed=3333, mode='real')
    r2 = pm_t.generate_whitenoise(seed=3333, mode='real')

    r1r = numpy.concatenate(comm.allgather(numpy.array(r1.ravel())))
    r2r = numpy.concatenate(comm.allgather(numpy.array(r2.ravel())))

    assert_array_equal(r1r, r2r)

@MPITest(commsize=(1,4))
def test_inplace_fft(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)
    numpy.random.seed(1234)
    if comm.rank == 0:
        Npar = 100
    else:
        Npar = 0

    pos = 1.0 * (numpy.arange(Npar * len(pm.Nmesh))).reshape(-1, len(pm.Nmesh)) * (7, 7)
    pos %= (pm.Nmesh + 1)
    layout = pm.decompose(pos)

    npos = layout.exchange(pos)

    real = pm.paint(npos)
    complex = real.r2c()
    complex2 = real.r2c(out=Ellipsis)

    assert real.base is complex2.base
    assert_almost_equal(numpy.asarray(complex), numpy.asarray(complex2), decimal=7)

    real = complex2.c2r()
    real2 = complex2.c2r(out=Ellipsis)
    assert real2.base is complex2.base
    assert_almost_equal(numpy.asarray(real), numpy.asarray(real2), decimal=7)

@MPITest(commsize=(1,4))
def test_c2c(comm):
    # this test requires pfft-python 0.1.16.

    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='complex128')
    numpy.random.seed(1234)
    if comm.rank == 0:
        Npar = 100
    else:
        Npar = 0

    pos = 1.0 * (numpy.arange(Npar * len(pm.Nmesh))).reshape(-1, len(pm.Nmesh)) * (7, 7)
    pos %= (pm.Nmesh + 1)
    layout = pm.decompose(pos)

    npos = layout.exchange(pos)
    real = pm.paint(npos)

    complex = real.r2c()

    real2 = complex.c2r()

    assert numpy.iscomplexobj(real)
    assert numpy.iscomplexobj(real2)
    assert numpy.iscomplexobj(complex)

    assert_array_equal(complex.cshape, pm.Nmesh)
    assert_array_equal(real2.cshape, pm.Nmesh)
    assert_array_equal(real.cshape, pm.Nmesh)

    real.readout(npos)
    assert_almost_equal(numpy.asarray(real), numpy.asarray(real2), decimal=7)

@MPITest(commsize=(1,4))
def test_decompose(comm):
    pm = ParticleMesh(BoxSize=4.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')
    numpy.random.seed(1234)
    if comm.rank == 0:
        Npar = 1000
    else:
        Npar = 0

    pos = 4.0 * (numpy.random.uniform(size=(Npar, 3)))

    pos = pm.generate_uniform_particle_grid()

    all_pos = numpy.concatenate(comm.allgather(pos), axis=0)

    for resampler in ['cic', 'tsc', 'db12']:
        def test(resampler):
            print(resampler)
            truth = numpy.zeros(pm.Nmesh, dtype='f8')
            affine = window.Affine(ndim=3, period=4)
            window.FindResampler(resampler).paint(truth, all_pos,
                transform=affine)
            truth = comm.bcast(truth)
            layout = pm.decompose(pos, smoothing=resampler)
            npos = layout.exchange(pos)
            real = pm.paint(npos, resampler=resampler)

            full = numpy.zeros(pm.Nmesh, dtype='f8')
            full[real.slices] = real
            full = comm.allreduce(full)
            #print(full.sum(), pm.Nmesh.prod())
            #print(truth.sum(), pm.Nmesh.prod())
            #print(comm.rank, npos, real.slices)
            assert_almost_equal(full, truth)
        # can't yield!
        test(resampler)

@MPITest(commsize=(1))
def test_indices(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')
    assert_almost_equal(pm.k[0], [[0], [0.785], [-1.571], [-0.785]], decimal=3)
    assert_almost_equal(pm.k[1], [[0, 0.785, -1.571]], decimal=3)
    assert_almost_equal(pm.x[0], [[0], [2], [-4], [-2]], decimal=3)
    assert_almost_equal(pm.x[1], [[0, 2, -4, -2]], decimal=3)

@MPITest(commsize=(1))
def test_indices_c2c(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='c16')
    assert_almost_equal(pm.k[0], [[0], [0.785], [-1.571], [-0.785]], decimal=3)
    assert_almost_equal(pm.k[1], [[0, 0.785, -1.571, -0.785]], decimal=3)
    assert_almost_equal(pm.x[0], [[0], [2], [-4], [-2]], decimal=3)
    assert_almost_equal(pm.x[1], [[0, 2, -4, -2]], decimal=3)

def assert_same_base(a1, a2):
    def find_base(a):
        base = a
        while getattr(base, 'base', None) is not None:
            base = base.base
        return base
    assert find_base(a1) is find_base(a2)

@MPITest(commsize=(1, 4))
def test_real_iter(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)

    for i, x, slab in zip(real.slabs.i, real.slabs.x, real.slabs):
        assert_same_base(slab, real.value)

        assert_almost_equal(slab.shape, sum(x[d] ** 2 for d in range(len(pm.Nmesh))).shape)
        for a, b in zip(slab.x, x):
            assert_array_equal(a, b)
        for a, b in zip(slab.i, i):
            assert_array_equal(a, b)
        # FIXME: test i!!

@MPITest(commsize=1)
def test_real_apply(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)
    def filter(x, v):
        xnormp = x.normp()
        assert_allclose(xnormp, sum(xi ** 2 for xi in x))
        return x[0] * 10 + x[1]

    real.apply(filter, out=Ellipsis)

    for i, x, slab in zip(real.slabs.i, real.slabs.x, real.slabs):
        assert_array_equal(slab, x[0] * 10 + x[1])

@MPITest(commsize=1)
def test_complex_apply(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    complex = ComplexField(pm)
    def filter(k, v):
        knormp = k.normp()
        assert_allclose(knormp, sum(ki ** 2 for ki in k))
        return k[0] + k[1] * 1j

    complex.apply(filter, out=Ellipsis)

    for i, x, slab in zip(complex.slabs.i, complex.slabs.x, complex.slabs):
        assert_array_equal(slab, x[0] + x[1] * 1j)

@MPITest(commsize=(1, 2, 3, 4))
def test_sort(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 6], comm=comm, dtype='f8')
    real = RealField(pm)
    truth = numpy.arange(8 * 6)
    real[...] = truth.reshape(8, 6)[real.slices]
    unsorted = real.copy()
    real.sort(out=Ellipsis)
    conjecture = numpy.concatenate(comm.allgather(real.value.ravel()))
    assert_array_equal(conjecture, truth)

    real.unravel(real)
    assert_array_equal(real, unsorted)

    complex = ComplexField(pm)
    truth = numpy.arange(8 * 4)
    complex[...] = truth.reshape(8, 4)[complex.slices]
    complex.ravel(out=Ellipsis)
    conjecture = numpy.concatenate(comm.allgather(complex.value.ravel()))
    assert_array_equal(conjecture, truth)

@MPITest(commsize=(1, 2, 3, 4))
def test_fdownsample(comm):
    """ fourier space resample, deprecated """
    pm1 = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    pm2 = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    numpy.random.seed(3333)
    truth = numpy.fft.rfftn(numpy.random.normal(size=(8, 8)))

    complex1 = ComplexField(pm1)
    for ind in numpy.ndindex(*complex1.cshape):
        complex1.csetitem(ind, truth[ind])

    assert_almost_equal(complex1[...], complex1.c2r().r2c())
    complex2 = ComplexField(pm2)
    for ind in numpy.ndindex(*complex2.cshape):
        newind = tuple([i if i <= 2 else 8 - (4 - i) for i in ind])
        if any(i == 2 for i in ind):
            complex2.csetitem(ind, 0)
        else:
            complex2.csetitem(ind, truth[newind])

    tmpr = RealField(pm2)
    tmp = ComplexField(pm2)

    complex1.resample(tmp)

    assert_almost_equal(complex2[...], tmp[...], decimal=5)

    complex1.c2r().resample(tmp)

    assert_almost_equal(complex2[...], tmp[...], decimal=5)

    complex1.resample(tmpr)

    assert_almost_equal(tmpr.r2c(), tmp[...])

    complex1.c2r().resample(tmpr)

    assert_almost_equal(tmpr.r2c(), tmp[...])

@MPITest(commsize=(1, 2, 3, 4))
def test_real_resample(comm):
    from functools import reduce

    pmh = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    pml = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    reall = pml.create(mode='real')
    reall.apply(lambda i, v: (i[0] % 2) * (i[1] %2 ), kind='index', out=Ellipsis)
    for resampler in ['nearest', 'cic', 'tsc', 'cubic']:
        realh = pmh.upsample(reall, resampler=resampler, keep_mean=False)
        reall2 = pml.downsample(realh, resampler=resampler)
    #    print(resampler, comm.rank, comm.size, reall, realh)
        assert_almost_equal(reall.csum(), realh.csum())
        assert_almost_equal(reall.csum(), reall2.csum())

@MPITest(commsize=(1, 2, 3, 4))
def test_cmean(comm):
    # this tests cmean (collective mean) along with resampling preseves it.

    pm1 = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    pm2 = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    complex1 = ComplexField(pm1)
    complex2 = ComplexField(pm2)
    real2 = RealField(pm2)
    real1 = RealField(pm1)
    for i, kk, slab in zip(complex1.slabs.i, complex1.slabs.x, complex1.slabs):
        slab[...] = sum([k**2 for k in kk]) **0.5

    complex1.c2r(real1)
    real1.resample(real2)
    assert_almost_equal(real1.cmean(), real2.cmean())

@MPITest(commsize=(1, 2, 3, 4))
def test_fupsample(comm):
    pm1 = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    pm2 = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')

    numpy.random.seed(3333)
    truth = numpy.fft.rfftn(numpy.random.normal(size=(8, 8)))

    complex1 = ComplexField(pm1)
    for ind in numpy.ndindex(*complex1.cshape):
        complex1.csetitem(ind, truth[ind])
        if any(i == 4 for i in ind):
            complex1.csetitem(ind, 0)
        else:
            complex1.csetitem(ind, truth[ind])

        if any(i >= 2 and i < 7 for i in ind):
            complex1.csetitem(ind, 0)

    assert_almost_equal(complex1[...], complex1.c2r().r2c())
    complex2 = ComplexField(pm2)
    for ind in numpy.ndindex(*complex2.cshape):
        newind = tuple([i if i <= 2 else 8 - (4 - i) for i in ind])
        if any(i == 2 for i in ind):
            complex2.csetitem(ind, 0)
        else:
            complex2.csetitem(ind, truth[newind])

    tmpr = RealField(pm1)
    tmp = ComplexField(pm1)

    complex2.resample(tmp)

    assert_almost_equal(complex1[...], tmp[...], decimal=5)

    complex2.c2r().resample(tmp)

    assert_almost_equal(complex1[...], tmp[...], decimal=5)

    complex2.resample(tmpr)

    assert_almost_equal(tmpr.r2c(), tmp[...])

    complex2.c2r().resample(tmpr)

    assert_almost_equal(tmpr.r2c(), tmp[...])

    
@MPITest(commsize=(1, 4))
def test_complex_iter(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    complex = ComplexField(pm)

    for x, slab in zip(complex.slabs.x, complex.slabs):
        assert_array_equal(slab.shape, sum(x[d] ** 2 for d in range(len(pm.Nmesh))).shape)
        for a, b in zip(slab.x, x):
            assert_almost_equal(a, b)

@MPITest(commsize=(1, 4))
def test_ctol(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')
    complex = ComplexField(pm)
    value, local = complex._ctol((3, 3))
    assert local is None

@MPITest(commsize=(1, 4))
def test_cgetitem(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4], comm=comm, dtype='f8')
    for i in numpy.ndindex((4, 4)):
        complex = RealField(pm)
        complex[...] = 0
        v2 = complex.csetitem(i, 100.)
        v1 = complex.cgetitem(i)
        assert v2 == 100.
        assert_array_equal(v1, v2)

    for i in numpy.ndindex((4, 3)):
        complex = ComplexField(pm)
        complex[...] = 0
        v2 = complex.csetitem(i, 100. + 10j)
        complex.c2r(out=Ellipsis).r2c(out=Ellipsis)
        v1 = complex.cgetitem(i)
        if i == (0, 0):
            assert v2 == 100.
            assert comm.allreduce(complex.value.sum()) == 100.
        elif i == (0, 2):
            assert v2 == 100.
            assert comm.allreduce(complex.value.sum()) == 100.
        elif i == (2, 0):
            assert v2 == 100.
            assert comm.allreduce(complex.value.sum()) == 100.
        elif i == (1, 0):
            assert v2 == 100 + 10j
            assert comm.allreduce(complex.value.sum()) == 200.
        elif i == (3, 0):
            assert v2 == 100 + 10j
            assert comm.allreduce(complex.value.sum()) == 200.
        elif i == (3, 2):
            assert v2 == 100 + 10j
            assert comm.allreduce(complex.value.sum()) == 200.
        elif i == (1, 2):
            assert v2 == 100 + 10j
            assert comm.allreduce(complex.value.sum()) == 200.
        elif i == (2, 2):
            assert v2 == 100.
            assert comm.allreduce(complex.value.sum()) == 100.
        else:
            assert v2 == 100. + 10j
            assert_array_equal(comm.allreduce(complex.value.sum()), 100. + 10j)
        assert_array_equal(v1, v2)

    for i in numpy.ndindex((4, 3, 2)):
        complex = ComplexField(pm)
        complex[...] = 0
        v2 = complex.csetitem(i, 100.)
        complex.c2r(out=Ellipsis).r2c(out=Ellipsis)
        v1 = complex.cgetitem(i)
        if i == (0, 0, 0):
            assert v2 == 100.
        if i == (0, 0, 1):
            assert v2 == 0.
        elif i == (0, 2, 0):
            assert v2 == 100.
        elif i == (0, 2, 1):
            assert v2 == 0.
        elif i == (2, 0, 0):
            assert v2 == 100.
        elif i == (2, 0, 1):
            assert v2 == 0.
        elif i == (2, 2, 0):
            assert v2 == 100.
        elif i == (2, 2, 1):
            assert v2 == 0.
        else:
            assert v2 == 100.
        assert_array_equal(v1, v2)

@MPITest(commsize=(1, 4))
def test_whitenoise(comm):
    # the whitenoise shall preserve the large scale.
    pm0 = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8, 8], comm=comm, dtype='f8')
    pm1 = ParticleMesh(BoxSize=8.0, Nmesh=[16, 16, 16], comm=comm, dtype='f8')
    pm2 = ParticleMesh(BoxSize=8.0, Nmesh=[32, 32, 32], comm=comm, dtype='f8')
    complex1_down = ComplexField(pm0)
    complex2_down = ComplexField(pm0)

    complex1 = pm1.generate_whitenoise(seed=8, unitary=True)
    complex2 = pm2.generate_whitenoise(seed=8, unitary=True)

    complex1.resample(complex1_down)
    complex2.resample(complex2_down)

    mask1 = complex1_down.value != complex2_down.value
    assert_array_equal(complex1_down.value, complex2_down.value)

@MPITest(commsize=(1, 4))
def test_whitenoise_mean(comm):
    # the whitenoise shall preserve the large scale.
    pm0 = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8, 8], comm=comm, dtype='f8')

    complex1 = pm0.generate_whitenoise(seed=8, unitary=True, mean=1.0)

    assert_allclose(complex1.c2r().cmean(), 1.0)

@MPITest(commsize=(1, 4))
def test_readout(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[8, 8], comm=comm, dtype='f8')
    real = RealField(pm)
    real.value[...] = 1.0
    pos = numpy.ones ((1, 2))
    out = numpy.empty((1), dtype='f8')
    real.readout(pos, out=out)

    pos = numpy.ones ((1, 2), dtype='f4')
    out = numpy.empty((1), dtype='f8')
    real.readout(pos, out=out)

    pos = numpy.ones ((1, 2), dtype='f4')
    out = numpy.empty((1), dtype='f4')
    real.readout(pos, out=out)

@MPITest(commsize=(1))
def test_cdot_cnorm(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')
    comp1 = pm.generate_whitenoise(1234, mode='complex')

    norm1 = comp1.cdot(comp1)
    norm2 = comp1.cnorm()
    norm3 = (abs(numpy.fft.fftn(numpy.fft.irfftn(comp1.value))) ** 2).sum()
    assert_allclose(norm2, norm3)
    assert_allclose(norm2, norm1)

@MPITest(commsize=(1))
def test_cnorm_log(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')
    comp1 = pm.generate_whitenoise(1234, mode='complex', mean=1.0)

    norm2 = comp1.cnorm(norm = lambda x: numpy.log(x.real ** 2 + x.imag ** 2))
    norm3 = (numpy.log(abs(numpy.fft.fftn(numpy.fft.irfftn(comp1.value))) ** 2)).sum()
    assert_allclose(norm2, norm3)

@MPITest(commsize=(1))
def test_cdot(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')
    comp1 = pm.generate_whitenoise(1234, mode='complex')
    comp2 = pm.generate_whitenoise(1239, mode='complex')

    norm1 = comp1.cdot(comp2)
    norm2 = comp2.cdot(comp1)

    norm_r = comp1.c2r().cdot(comp2.c2r()) / pm.Nmesh.prod()

    assert_allclose(norm2.real, norm_r)
    assert_allclose(norm1.real, norm_r)
    assert_allclose(norm1.real, norm2.real)
    assert_allclose(norm1.imag, -norm2.imag)

@MPITest(commsize=(1))
def test_cdot_c2c(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='c16')
    comp1 = pm.generate_whitenoise(1234, mode='complex')
    comp2 = pm.generate_whitenoise(1239, mode='complex')

    norm1 = comp1.cdot(comp2)
    norm2 = comp2.cdot(comp1)

    r1 = comp1.c2r()

    norm_r = comp1.c2r().cdot(comp2.c2r()) / pm.Nmesh.prod()

    assert_allclose(norm2.real, norm_r)
    assert_allclose(norm1.real, norm_r)
    assert_allclose(norm1.real, norm2.real)
    assert_allclose(norm1.imag, -norm2.imag)

@MPITest(commsize=(1, 4))
def test_preview(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')

    comp1 = pm.generate_whitenoise(1234, mode='real')

    preview = comp1.preview(axes=(0, 1, 2))

    preview = comp1.preview(Nmesh=4, axes=(0, 1, 2))

    for ind1 in numpy.ndindex(*(list(comp1.cshape))):
        assert_allclose(preview[ind1], comp1.cgetitem(ind1))

    preview1 = comp1.preview(Nmesh=4, axes=(0, 1))
    previewsum1 = preview.sum(axis=2)
    assert_allclose(preview1, previewsum1)

    preview2 = comp1.preview(Nmesh=4, axes=(1, 2))
    previewsum2 = preview.sum(axis=0)
    assert_allclose(preview2, previewsum2)

    preview3 = comp1.preview(Nmesh=4, axes=(0, 2))
    previewsum3 = preview.sum(axis=1)
    assert_allclose(preview3, previewsum3)

    preview4 = comp1.preview(Nmesh=4, axes=(2, 0))
    previewsum4 = preview.sum(axis=1).T
    assert_allclose(preview4, previewsum4)

    preview5 = comp1.preview(Nmesh=4, axes=(0,))
    previewsum5 = preview.sum(axis=(1, 2))
    assert_allclose(preview5, previewsum5)

    preview6 = comp1.preview(Nmesh=8, axes=(0,))

@MPITest(commsize=(1, 4))
def test_c2c_r2c_edges(comm):
    pm1 = ParticleMesh(BoxSize=8.0, Nmesh=[5, 7, 9], comm=comm, dtype='c16')
    pm2 = ParticleMesh(BoxSize=8.0, Nmesh=[5, 7, 9], comm=comm, dtype='f8')

    assert_allclose(pm1.x[0], pm2.x[0])
    assert_allclose(pm1.x[1], pm2.x[1])
    assert_allclose(pm1.x[2], pm2.x[2])

@MPITest(commsize=(1))
def test_grid(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')
    grid = pm.generate_uniform_particle_grid(shift=0.5)
    assert_array_equal(pm.comm.allreduce(grid.shape[0]), pm.Nmesh.prod())
    real = pm.paint(grid)
    assert_array_equal(real, 1.0)

@MPITest(commsize=(1, 4))
def test_grid(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')
    grid, id = pm.generate_uniform_particle_grid(shift=0.5, return_id=True)
    assert len(id) == len(grid)

    allid = numpy.concatenate(comm.allgather(id), axis=0)
    # must be all unique
    assert len(numpy.unique(allid)) == len(allid)
    assert numpy.max(allid) == len(allid) - 1
    assert numpy.min(allid) == 0

@MPITest(commsize=(1, 4))
def test_grid_shifted(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=[4, 4, 4], comm=comm, dtype='f8')
    grid = pm.generate_uniform_particle_grid(shift=0.5)
    grid = grid + 4.0
    assert_array_equal(pm.comm.allreduce(grid.shape[0]), pm.Nmesh.prod())

    layout = pm.decompose(grid)
    g2 = layout.exchange(grid)
    real = pm.paint(grid, layout=layout)
    #print(real, g2 % 8, real.slices)
    assert_allclose(real, 1.0)

    grid = grid - 6.1
    assert_array_equal(pm.comm.allreduce(grid.shape[0]), pm.Nmesh.prod())

    layout = pm.decompose(grid)
    real = pm.paint(grid, layout=layout)
    assert_allclose(real, 1.0)

@MPITest(commsize=(1))
def test_leak(comm):
    # 1024 is long enough to crash MPICH.
    a = []
    for i in range(1024):
        a.append(ParticleMesh(BoxSize=8.0, Nmesh=[128, 128, 128], comm=comm, dtype='f8'))
