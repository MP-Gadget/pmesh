import numpy
from numpy.testing import assert_array_equal, assert_allclose, assert_almost_equal
from pmesh.invariant import get_index
import pytest

def test_1d():
    x = numpy.arange(-4, 5).reshape(-1, 1)
    ind = get_index(x, 6, compressed=False)
    # some are out of bound
    assert_array_equal(ind, [-1, 5, 4, 2, 0, 1, 3, 5, -1])

def test_1dc():
    x = numpy.arange(-4, 5).reshape(-1, 1)
    ind = get_index(x, 6, compressed=True)
    # some are out of bound
    assert_array_equal(ind, [ -1, 3, -1, -1, 0, 1, 2, 3, -1])

@pytest.mark.parametrize('pos', [False, True])
def test_2d(pos):
    x1 = numpy.arange(-2, 2)
    if pos:
        x1[0] *= -1 # test using positive freq
    x = numpy.empty((4, 4, 2), dtype='int')
    x[..., 0] = x1.reshape(1, -1)
    x[..., 1] = x1.reshape(-1, 1)

    ind = get_index(x, 4, compressed=False)
    assert_array_equal(ind,
             [[15, 14, 12, 13],
              [11,  8,  6,  7],
              [ 9,  4,  0,  1],
              [10,  5,  2,  3]]
    )

@pytest.mark.parametrize('pos', [False, True])
def test_2dc(pos):
    x1 = numpy.arange(-2, 2)
    if pos:
        x1[0] *= -1
    x = numpy.empty((4, 4, 2), dtype='int')
    x[..., 0] = x1.reshape(1, -1)
    x[..., 1] = x1.reshape(-1, 1)

    ind = get_index(x, 4, compressed=True)
    assert_array_equal(ind,
        [[11, 10,  8,  9],
              [-1, -1, -1, -1],
              [ 6,  4,  0,  1],
              [ 7,  5,  2,  3]]
    )

@pytest.mark.parametrize('pos, c', [(False, False), (True, False), (True, True), (False, True)])
def test_3d(pos, c):
    x1 = numpy.arange(-3, 3)
    if pos:
        x1[0] *= -1 # test using positive freq
    x = numpy.empty((6, 6, 6, 3), dtype='int')
    x[..., 0] = x1.reshape(1, 1, -1)
    x[..., 1] = x1.reshape(1, -1, 1)
    x[..., 2] = x1.reshape(-1, 1, 1)

    x = x.reshape(-1, 3)
    indm = get_index(x, 6, compressed=c, maxlength=10)
    assert (indm < 10).all()

    ind = get_index(x, 6, compressed=c)
    if c:
        assert_array_equal(
            ind[(abs(x) != 3).all(axis=-1) & (x[..., 2] < 0)], -1)
        mask = ind >= 0
    else:
        mask = ind == ind
    for cut in range(1, ind[mask].max() - 1):
        inside = abs(x[mask & (ind < cut)]).max(axis=-1).max()
        outside = abs(x[mask & (ind >= cut)]).max(axis=-1).min()
        assert inside <= outside
    if c:
        assert ind[mask].max() == 6**2 * 4 - 1
    else:
        assert ind[mask].max() == 6**3 - 1
    assert ind[mask].min() == 0

