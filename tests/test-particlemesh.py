from pypm.particlemesh import ParticleMesh
from numpy.testing import assert_allclose
def test_roundtrip_normalization():
    pm = ParticleMesh(10.0, 2)
    pm.real[:] = 1.0
    pm.r2c()
    assert_allclose(abs(pm.complex[0, 0, 0]), 1.0 * pm.BoxSize.prod())
    pm.c2r()
    assert_allclose(pm.real, 1.0)

    
