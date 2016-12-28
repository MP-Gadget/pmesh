import numpy
from numpy.testing import assert_array_equal, assert_allclose, assert_almost_equal
from numpy.testing.decorators import skipif

from pmesh.whitenoise import generate

def test_generate():
    value = numpy.zeros((64, 64, 64), dtype='complex128')

    generate(value, 0, value.shape, 1)

    piece = numpy.zeros((32, 4, 4),dtype='complex128')
    offset = [2, 2, 2]
    offset = [2, 2, 2]
    generate(piece, offset, value.shape, 1)
    truth = value[
        offset[0]:offset[0] + piece.shape[0],
        offset[1]:offset[1] + piece.shape[1],
        offset[2]:offset[2] + piece.shape[2]]

    assert_array_equal(piece, truth)

def test_generate_2d():
    value = numpy.zeros((64, 64), dtype='complex128')

    generate(value, 0, value.shape, 1)

    piece = numpy.zeros((32, 4),dtype='complex128')
    offset = [2, 2]
    offset = [2, 2]
    generate(piece, offset, value.shape, 1)
    truth = value[
        offset[0]:offset[0] + piece.shape[0],
        offset[1]:offset[1] + piece.shape[1]]

    assert_array_equal(piece, truth)

def test_generate_1d():
    value = numpy.zeros((64), dtype='complex128')

    generate(value, 0, value.shape, 1)

    piece = numpy.zeros((32),dtype='complex128')
    offset = [2]
    offset = [2]
    generate(piece, offset, value.shape, 1)
    truth = value[
        offset[0]:offset[0] + piece.shape[0]]

    assert_array_equal(piece, truth)
