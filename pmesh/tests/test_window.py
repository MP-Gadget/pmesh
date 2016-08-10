from pmesh.window import WindowResampler

import numpy
from numpy.testing import assert_array_equal, assert_allclose

def test_unweighted():
    wcic = WindowResampler("linear", 2, ndim=2)

    real = numpy.zeros((4, 4))
    pos = [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]
    wcic.paint(real, pos)

    assert_array_equal(real,
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])

def test_weighted():
    wcic = WindowResampler("linear", 2, ndim=2)

    real = numpy.zeros((4, 4))
    pos = [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
    ]
    mass = [0, 1, 2, 3]
    wcic.paint(real, pos, mass)
    assert_array_equal(real,
        [[0, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 3]])

def test_wide():
    wcic = WindowResampler("linear", 4, ndim=2)
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
    wcic = WindowResampler("linear", 2, ndim=2, period=2)
    real = numpy.zeros((2, 2))
    pos = [
        [-.5, -.5],
    ]
    wcic.paint(real, pos)
    assert_array_equal(real,
        [[0.25, 0.25],
         [0.25, 0.25]])

    real = numpy.zeros((2, 2))
    pos = [
        [-.5, .5],
    ]
    wcic.paint(real, pos)
    assert_array_equal(real,
        [[0.25, 0.25],
         [0.25, 0.25]])

    real = numpy.zeros((2, 2))
    pos = [
        [-.5, 1.5],
    ]
    wcic.paint(real, pos)
    assert_array_equal(real,
        [[0.25, 0.25],
         [0.25, 0.25]])

def test_translate():
    wcic = WindowResampler("linear", 2, ndim=2, translate=[-1, 0])
    real = numpy.zeros((2, 2))
    pos = [
        [1., 0],
    ]
    wcic.paint(real, pos)
    assert_array_equal(real,
        [[1., 0.],
         [0., 0.]])

def test_scale():
    wcic = WindowResampler("linear", 2, ndim=2, translate=[-1, 0], scale=0.1)
    real = numpy.zeros((2, 2))
    pos = [
        [10., 0],
    ]
    wcic.paint(real, pos)
    assert_array_equal(real,
        [[1., 0.],
         [0., 0.]])


def test_strides():
    wcic = WindowResampler("linear", 2, ndim=2)
    real = numpy.zeros((20, 20))[::10, ::10]
    pos = [
        [1., 0],
    ]
    wcic.paint(real, pos)
    assert_array_equal(real,
        [[0, 0],
         [1, 0]])

def test_anisotropic():
    wcic = WindowResampler("linear", 2, ndim=2)
    real = numpy.zeros((2, 4))
    pos = [
        [0., 0],
        [1., 0],
        [0., 1],
        [0., 2],
        [0., 3],
    ]
    wcic.paint(real, pos)
    assert_array_equal(real,
        [[1, 1, 1, 1],
         [1, 0, 0, 0]])

def test_diff():
    wcic = WindowResampler("linear", 2, ndim=2)
    real = numpy.zeros((2, 2))
    pos = [
        [0., 0],
    ]
    wcic.paint(real, pos, diffdir=0)
    assert_array_equal(real,
        [[-1, 0],
         [1, 0]])

    real = numpy.zeros((2, 2))
    wcic.paint(real, pos, diffdir=1)
    assert_array_equal(real,
        [[-1, 1],
         [0, 0]])
