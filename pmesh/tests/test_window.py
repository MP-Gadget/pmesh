from pmesh.window import ResampleWindow, Affine, CIC, LANCZOS2, TSC, CUBIC, DB12, DB20

import numpy
from numpy.testing import assert_array_equal, assert_allclose, assert_almost_equal

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
    real = numpy.zeros((4, 4))
    pos = [
        [1.5, 1.5],
    ]
    wcic.paint(real, pos)
    assert_array_equal(real,
    [[ 0.015625, 0.046875, 0.046875, 0.015625],
     [ 0.046875, 0.140625, 0.140625, 0.046875],
     [ 0.046875, 0.140625, 0.140625, 0.046875],
     [ 0.015625, 0.046875, 0.046875, 0.015625]])


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
        [0., 0],
    ]
    CIC.paint(real, pos, diffdir=0)
    assert_array_equal(real,
        [[-1, 0],
         [1, 0]])

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
    [[ 0.003906, -0.035156, -0.035156,  0.003906],
     [-0.035156,  0.316406,  0.316406, -0.035156],
     [-0.035156,  0.316406,  0.316406, -0.035156],
     [ 0.003906, -0.035156, -0.035156,  0.003906]], atol=1e-5)

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
    real = numpy.zeros((4))
    pos = [
        [1.5],
    ]
    CUBIC.paint(real, pos)
    assert_array_equal(real, [-0.0625, 0.5625, 0.5625, -0.0625])

def test_db12():
    real = numpy.zeros((8))
    pos = [
        [3.0],
    ]
    DB12.paint(real, pos)
    assert_almost_equal(real, [0., 0.1552735, -0.3257409, 0.8702795 , 0.2936153, 0.0065725, 0., 0. ] )

def test_db20():
    real = numpy.zeros((8))
    pos = [
        [3.0],
    ]
    DB20.paint(real, pos)
    assert_almost_equal(real, [  2.896390e-01,  -4.214947e-01,   4.631021e-01,   5.777651e-01,
         8.876256e-02,   2.223313e-03,   2.640220e-06,   0.000000e+00]);
