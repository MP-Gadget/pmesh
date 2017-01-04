import numpy
from numpy.testing import assert_array_equal, assert_allclose, assert_almost_equal
from numpy.testing.decorators import skipif

from pmesh.whitenoise import generate

def test_generate_3d():
    Nmesh = 128
    value = numpy.zeros((Nmesh, Nmesh, Nmesh//2 + 1), dtype='complex128')

    generate(value, 0, (Nmesh, Nmesh, Nmesh), 1, unitary=False)
    assert_allclose(value.real.std(), 0.5 ** 0.5, rtol=1e-2)
    assert_allclose(value.imag.std(), 0.5 ** 0.5, rtol=1e-2)

    piece = numpy.zeros((32, 4, 4),dtype='complex128')
    offset = [2, 2, 2]
    offset = [2, 2, 2]
    generate(piece, offset, (Nmesh, Nmesh, Nmesh), 1, unitary=False)
    truth = value[
        offset[0]:offset[0] + piece.shape[0],
        offset[1]:offset[1] + piece.shape[1],
        offset[2]:offset[2] + piece.shape[2]]

    assert_array_equal(piece, truth)

def test_generate_2d():
    Nmesh = 1024
    value = numpy.zeros((Nmesh, Nmesh//2 + 1), dtype='complex128')

    generate(value, 0, (Nmesh, Nmesh), 1, unitary=False)
    assert_allclose(value.real.std(), 0.5 ** 0.5, rtol=1e-1)
    assert_allclose(value.imag.std(), 0.5 ** 0.5, rtol=1e-1)

    piece = numpy.zeros((32, 4),dtype='complex128')
    offset = [2, 2]
    generate(piece, offset, (Nmesh, Nmesh), 1, unitary=False)
    truth = value[
        offset[0]:offset[0] + piece.shape[0],
        offset[1]:offset[1] + piece.shape[1]]

    assert_array_equal(piece, truth)

def test_generate_1d():
    Nmesh = 4096 * 32
    value = numpy.zeros((Nmesh//2 + 1), dtype='complex128')

    generate(value, 0, (Nmesh,), 1, unitary=False)
    assert_allclose(value.real.std(), 0.5 ** 0.5, rtol=1e-1)
    assert_allclose(value.imag.std(), 0.5 ** 0.5, rtol=1e-1)

    piece = numpy.zeros((8),dtype='complex128')
    offset = [2]
    offset = [2]
    generate(piece, offset, (Nmesh,), 1, unitary=False)
    truth = value[
        offset[0]:offset[0] + piece.shape[0]]

    assert_array_equal(piece, truth)
