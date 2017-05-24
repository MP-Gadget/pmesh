from pmesh.window import ResampleWindow, Affine, CIC, LANCZOS2, TSC, QUADRATIC, CUBIC, DB12, DB20, LINEAR

import numpy
from numpy.testing import assert_array_equal, assert_allclose, assert_almost_equal
from numpy.testing.decorators import skipif

def test_unweighted():
    real = numpy.zeros((4, 4))
    pos = [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]
    CIC.paint(real, pos)

    assert_array_equal(real,
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])

def test_weighted():
    real = numpy.zeros((4, 4))
    pos = [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]
    mass = [0, 1, 2, 3]
    CIC.paint(real, pos, mass)
    assert_array_equal(real,
        [[0, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 3]])

def test_wide():
    wcic = ResampleWindow("linear", 4)
    real = numpy.zeros((4))
    pos = [
        [1.5],
    ]
    wcic.paint(real, pos)
    assert_almost_equal(real, [ 0.125,  0.375,  0.375,  0.125])

    real = numpy.zeros((4))
    wcic.paint(real, [[1.51]])
    assert_almost_equal(real, [ 0.1225,  0.3725,  0.3775,  0.1275 ])

    real = numpy.zeros((4))
    wcic.paint(real, pos, diffdir=0)
    assert_almost_equal(real, [-0.25, -0.25, 0.25, 0.25])

def test_wrap():
    affine = Affine(ndim=2, period=2)
    real = numpy.zeros((2, 2))
    pos = [
        [-.5, -.5],
    ]
    CIC.paint(real, pos, transform=affine)
    assert_array_equal(real,
        [[0.25, 0.25],
         [0.25, 0.25]])

    real = numpy.zeros((2, 2))
    pos = [
        [-.5, .5],
    ]
    CIC.paint(real, pos, transform=affine)
    assert_array_equal(real,
        [[0.25, 0.25],
         [0.25, 0.25]])

    real = numpy.zeros((2, 2))
    pos = [
        [-.5, 1.5],
    ]
    CIC.paint(real, pos, transform=affine)
    assert_array_equal(real,
        [[0.25, 0.25],
         [0.25, 0.25]])

def test_translate():
    affine = Affine(ndim=2, translate=[-1, 0])
    real = numpy.zeros((2, 2))
    pos = [
        [1., 0],
    ]
    CIC.paint(real, pos, transform=affine)
    assert_array_equal(real,
        [[1., 0.],
         [0., 0.]])

def test_affine():
    affine = Affine(ndim=2)
    real = numpy.zeros((4, 4))
    pos = [
        [.5, .5],
    ]
    CIC.paint(real, pos, transform=affine)

    translate = numpy.zeros((4, 4))
    pos = [
        [0., 0.],
    ]
    shift = affine.shift

    CIC.paint(translate, pos, transform=affine.shift(0.5))
    assert_array_equal(translate, real)


def test_scale():
    affine = Affine(ndim=2, translate=[-1, 0], scale=0.1)
    real = numpy.zeros((2, 2))
    pos = [
        [10., 0],
    ]
    CIC.paint(real, pos, transform=affine)
    assert_array_equal(real,
        [[1., 0.],
         [0., 0.]])


def test_strides():
    real = numpy.zeros((20, 20))[::10, ::10]
    pos = [
        [1., 0],
    ]
    CIC.paint(real, pos)
    assert_array_equal(real,
        [[0, 0],
         [1, 0]])

def test_anisotropic():
    real = numpy.zeros((2, 4))
    pos = [
        [0., 0],
        [1., 0],
        [0., 1],
        [0., 2],
        [0., 3],
    ]
    CIC.paint(real, pos)
    assert_array_equal(real,
        [[1, 1, 1, 1],
         [1, 0, 0, 0]])

def test_diff():
    real = numpy.zeros((2, 2))
    pos = [
        [0.5, 0],
    ]
    CIC.paint(real, pos, diffdir=0)
    assert_array_equal(real,
        [[-1, 0],
         [1, 0]])

    pos = [
        [0, 0.5],
    ]
    real = numpy.zeros((2, 2))
    CIC.paint(real, pos, diffdir=1)
    assert_array_equal(real,
        [[-1, 1],
         [0, 0]])

def test_lanczos2():
    real = numpy.zeros((4, 4))
    pos = [
        [1.5, 1.5],
    ]
    LANCZOS2.paint(real, pos)
    assert_allclose(real,
      [[ 0.003977, -0.035797, -0.035797,  0.003977], 
       [-0.035797,  0.322173,  0.322173, -0.035797],
       [-0.035797,  0.322173,  0.322173, -0.035797],
       [ 0.003977, -0.035797, -0.035797,  0.003977]], atol=1e-5)
    assert_array_equal(LANCZOS2.support, 4)

def test_tsc():
    real = numpy.zeros((4))
    pos = [
        [1.5],
    ]
    TSC.paint(real, pos)
    assert_array_equal(real, [0, 0.5, 0.5, 0])

    real = numpy.zeros((4))
    pos = [
        [1.8],
    ]
    TSC.paint(real, pos)
    # this is special for odd support kernels. #10
    assert_almost_equal(real, [ 0.   ,  0.245,  0.71 ,  0.045])

    real = numpy.zeros((5))
    pos = [
        [2],
    ]
    TSC.paint(real, pos)
    assert_array_equal(real, [0, 0.125, 0.75, 0.125, 0])

    real = numpy.zeros((5))
    pos = [
        [0],
    ]
    affine = Affine(ndim=1, period=5)
    TSC.paint(real, pos, transform=affine)
    assert_array_equal(real, [0.75, 0.125, 0, 0, 0.125])

def test_cubic():
    real = numpy.zeros((6))
    pos = [
        [2.5],
    ]
    CUBIC.paint(real, pos)
    assert_array_equal(real, [0, -0.0625, 0.5625, 0.5625, -0.0625, 0])


@skipif(True, "numerical details of wavelets undecided")
def test_db12():
    real = numpy.zeros((10))
    pos = [
        [4.5],
    ]
    DB12.paint(real, pos)
    assert_almost_equal(real,  [ -8.0292000e-04,  -5.4008900e-03,   3.2057780e-02,  -9.6184070e-02,
         2.0444605e-01,  -3.4209320e-01,   3.9515054e-01,   7.4167734e-01,
         7.0450990e-02,   1.3687000e-04])

@skipif(True, "numerical details of wavelets undecided")
def test_db20():
    real = numpy.zeros((13))
    pos = [
        [6],
    ]
    DB20.paint(real, pos)
    assert_almost_equal(real, [  8.7396600e-03,  -6.7160500e-03,  -6.5716800e-03,   3.5024950e-02,
        -7.4218390e-02,   1.0433001e-01,  -7.1827390e-02,  -1.6736320e-01,
         8.4381209e-01,   3.1778939e-01,   2.0722960e-02,   1.5644000e-04,
         0.0000000e+00])

def test_cic_tuned():
    assert CIC.support == 2
    assert LINEAR.support == 2
    real = numpy.zeros((4, 4, 4))
    pos = [
        [1.1, 1.3, 2.5],
    ]
    CIC.paint(real, pos)

    real2 = numpy.zeros((4, 4, 4))
    LINEAR.paint(real2, pos)

    assert_array_equal(real, real2)

    for d in range(3):
        d1 = numpy.zeros((4, 4, 4))
        d2 = numpy.zeros((4, 4, 4))
        CIC.paint(d1, pos, diffdir=d)
        LINEAR.paint(d2, pos, diffdir=d)
        assert_array_equal(d1, d2)

def test_tsc_tuned():
    affine = Affine(ndim=3, translate=[2, 1, 2], scale=[0.5, 2.0, 1.1], period=[8, 8, 8])
    assert TSC.support == 3
    assert QUADRATIC.support == 3
    real = numpy.zeros((8, 8, 8))
    real2 = numpy.zeros((8, 8, 8))
    numpy.random.seed(1234)
    field = numpy.random.uniform(size=real.shape)

    pos = [
        [1.1, 1.3, 2.9],
    ]
    TSC.paint(real, pos, transform=affine)
    QUADRATIC.paint(real2, pos, transform=affine)
    v = TSC.readout(field, pos, transform=affine)
    v2 = QUADRATIC.readout(field, pos, transform=affine)

    assert_array_equal(real, real2)
    assert_array_equal(v, v2)

    for d in range(3):
        d1 = numpy.zeros((8, 8, 8))
        d2 = numpy.zeros((8, 8, 8))
        TSC.paint(d1, pos, diffdir=d, transform=affine)
        QUADRATIC.paint(d2, pos, diffdir=d, transform=affine)
        v = TSC.readout(field, pos, transform=affine)
        v2 = QUADRATIC.readout(field, pos, transform=affine)
        assert_array_equal(d1, d2)
        assert_array_equal(v, v2)
