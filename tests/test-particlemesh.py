from pypm.particlemesh import ParticleMesh
from numpy.testing import assert_allclose

def test_roundtrip_normalization():
    pm = ParticleMesh(10.0, 2)
    pm.real[:] = 1.0
    pm.r2c()
    assert_allclose(abs(pm.complex[0, 0, 0]), 1.0)
    pm.c2r()
    assert_allclose(pm.real, 1.0)

    
def test_xkrw():
    pm = ParticleMesh(10.0, 2)

    for i in range(3):
        assert(pm.r[i].shape[i] == 2)
        assert(pm.x[i].shape[i] == 2)

    for i in range(3):
        # it happends for 2x2x2 the last dim shall also
        # be 2 // 2 + 1 == 2
        assert(pm.k[i].shape[i] == 2)
        assert(pm.w[i].shape[i] == 2)
