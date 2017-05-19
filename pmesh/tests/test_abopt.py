from __future__ import print_function
from abopt.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger

from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
import numpy
import logging

logger.setLevel(level=logging.WARNING)

try:
    from abopt.engines.pmesh import ParticleMeshEngine, ParticleMesh, RealField, ComplexField, check_grad
    pm = ParticleMesh(BoxSize=1.0, Nmesh=(4, 4), dtype='f8', resampler='tsc')
except ImportError:
    pm = None

@skipif(pm == None)
def test_compute():
    def transfer(k): return 2.0
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.transfer(complex='c', tf=transfer)
    code.c2r(complex='c', real='r')
    code.norm(field='r', r='sum', metric=None)

    field = pm.generate_whitenoise(seed=1234).c2r()

    norm = code.compute('sum', init={'r': field})
    assert_allclose(norm, field.cnorm() * 4)

@skipif(pm == None)
def test_gradient():
    def transfer(k):
        k2 = sum(ki **2 for ki in k)
        k2[k2 == 0] = 1.0
#        return 1 / k2
        return 2.0
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.transfer(complex='c', tf=transfer)
    code.c2r(complex='c', real='r')
    code.norm(field='r', r='sum', metric=None)

    field = pm.generate_whitenoise(seed=1234).c2r()

    norm = code.compute('sum', init={'r': field})
    assert_allclose(norm, field.cnorm() * 4)
    norm, _r = code.compute_with_gradient(('sum', '_r'), init={'r': field}, ginit={'_sum': 1.0})
    assert_allclose(_r, field * 4 * 2)

@skipif(pm == None)

@skipif(pm == None)
def test_paint():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    numpy.random.seed(1234)
    s = numpy.random.uniform(size=engine.q.shape) * 0.1

    code.decompose(s='s', layout='layout')
    code.paint(s='s', mesh='density', layout='layout')

    check_grad(code, 'density', 's', init={'s': s}, eps=1e-4, rtol=1e-2)

@skipif(pm == None)
def test_readout():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    numpy.random.seed(1234)
    s = numpy.random.uniform(size=engine.q.shape) * 0.1

    field = pm.generate_whitenoise(seed=1234).c2r()

    code.decompose(s='s', layout='layout')
    code.readout(s='s', mesh='density', layout='layout', value='value')

    check_grad(code, 'value', 's', init={'density' : field, 's': s}, eps=1e-4, rtol=1e-2)

    check_grad(code, 'value', 'density', init={'density' : field, 's': s}, eps=1e-4, rtol=1e-2)

