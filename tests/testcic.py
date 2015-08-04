import numpy
from pypm import cic

def test_paint1():
    mesh = numpy.zeros((2, 2))
    pos = [[-.1, 0.0]]
    cic.paint(pos, mesh, period=2.0)
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0. ],
             [ 0.1,  0. ]]
            )

def test_paint2():
    mesh = numpy.zeros((2, 2))
    pos = [[.1, 0.0]]
    cic.paint(pos, mesh, period=2.0)
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0. ],
             [ 0.1,  0. ]]
            )

def test_paint3():
    mesh = numpy.zeros((2, 2))
    pos = [[0.0, 0.1]]
    cic.paint(pos, mesh, period=2.0)
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0.1 ],
             [ 0.0,  0. ]]
            )

def test_paint4():
    mesh = numpy.zeros((2, 2))
    pos = [[0.0, -0.1]]
    cic.paint(pos, mesh, period=2.0)
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0.1 ],
             [ 0.0,  0. ]]
            )

def test_paint5():
    mesh = numpy.zeros((2, 2))
    pos = [[1.1, 0.0]]
    cic.paint(pos, mesh, period=2.0)
    assert numpy.allclose(
            mesh, 
            [[ 0.1,  0.0 ],
             [ 0.9,  0. ]]
            )

def test_paint6():
    mesh = numpy.zeros((2, 2))
    pos = [[1.1, 2.0]]
    cic.paint(pos, mesh, period=2.0)
    assert numpy.allclose(
            mesh, 
            [[ 0.1,  0.0 ],
             [ 0.9,  0. ]]
            )
def test_paint1():
    mesh = numpy.zeros((2, 2))
    pos = [[-.1, 0.0]]
    cic.paint(pos, mesh, period=2.0)
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0. ],
             [ 0.1,  0. ]]
            )


def test_paintraise1():
    mesh = numpy.zeros((2, 2))
    pos = [[0.1, 0.0]]
    cic.paint(pos, mesh, mode='raise')
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0.0 ],
             [ 0.1,  0. ]]
            )

def test_paintraise2():
    mesh = numpy.zeros((2, 2))
    pos = [[0.0, 0.0]]
    cic.paint(pos, mesh, mode='raise')
    assert numpy.allclose(
            mesh, 
            [[ 1.0,  0.0 ],
             [ 0.0,  0. ]]
            )

def test_paintraise3():
    mesh = numpy.zeros((2, 2))
    pos = [[-.1, 0.0]]
    try:
        cic.paint(pos, mesh, mode='raise')
        raise AssertionError("shall not reach here")
    except ValueError as e:
        pass

def test_paintraise4():
    mesh = numpy.zeros((2, 2))
    pos = [[2.1, 0.0]]
    try:
        cic.paint(pos, mesh, mode='raise')
        raise AssertionError("shall not reach here")
    except ValueError as e:
        pass

def test_paintignore1():
    mesh = numpy.zeros((2, 2))
    pos = [[0.1, 0.0]]
    cic.paint(pos, mesh, mode='ignore')
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0.0 ],
             [ 0.1,  0. ]]
            )
    mesh[:] = 0
    cic.paint(pos, mesh, mode='ignore', period=4)
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0.0 ],
             [ 0.1,  0. ]]
            )

def test_paintignore2():
    mesh = numpy.zeros((2, 2))
    pos = [[0.0, 0.0]]
    cic.paint(pos, mesh, mode='ignore')
    assert numpy.allclose(
            mesh, 
            [[ 1.0,  0.0 ],
             [ 0.0,  0. ]]
            )
    mesh[:] = 0
    cic.paint(pos, mesh, mode='ignore', period=4)
    assert numpy.allclose(
            mesh, 
            [[ 1.0,  0.0 ],
             [ 0.0,  0. ]]
            )

def test_paintignore3():
    mesh = numpy.zeros((2, 2))
    pos = [[-.1, 0.0]]
    cic.paint(pos, mesh, mode='ignore')
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0.0 ],
             [ 0.0,  0. ]]
            )
    mesh[:] = 0
    cic.paint(pos, mesh, mode='ignore', period=4)
    assert numpy.allclose(
            mesh, 
            [[ 0.9,  0.0 ],
             [ 0.0,  0. ]]
            )

def test_paintignore4():
    mesh = numpy.zeros((2, 2))
    pos = [[2.1, 0.0]]
    cic.paint(pos, mesh, mode='ignore')
    assert numpy.allclose(
            mesh, 
            [[ 0.0,  0.0 ],
             [ 0.0,  0. ]]
            )
    mesh[:] = 0
    cic.paint(pos, mesh, mode='ignore', period=4)
    assert numpy.allclose(
            mesh, 
            [[ 0.0,  0.0 ],
             [ 0.0,  0. ]]
            )


test_paint1()
test_paint2()
test_paint3()
test_paint4()
test_paint5()
test_paint6()
test_paint5()
test_paintraise1()
test_paintraise2()
test_paintraise3()
test_paintraise4()
test_paintignore1()
test_paintignore2()
test_paintignore3()
test_paintignore4()

def test_readout1():
    mesh = numpy.zeros((2, 2))
    pos = [[-.1, 0.0]]
    mesh = numpy.array([
        [1., 1.],
        [1., 1.]])
    values = cic.readout(mesh, pos, period=2.0)
    assert numpy.allclose(values, 1.0)

def test_readout2():
    mesh = numpy.zeros((2, 2))
    pos = [[-.1, 0.0]]
    mesh = numpy.array([
        [1., 1.],
        [0., 1.]])
    values = cic.readout(mesh, pos, period=2.0)
    assert numpy.allclose(values, 0.9)

def test_readout3():
    mesh = numpy.zeros((2, 2))
    pos = [[-1.1, 0.0]]
    mesh = numpy.array([
        [1., 1.],
        [1., 1.]])
    values = cic.readout(mesh, pos, period=4.0, mode='ignore')
    assert numpy.allclose(values, 0.0)


#test_readout1()
#test_readout2()
#test_readout3()

from pypm.tools import Timers
def test_speed():
    t = Timers()
    mesh = numpy.zeros((100, 100, 100))
    pos = numpy.random.random(size=(1000000, 3))
    transform = lambda pos: pos * 100.
    with t['jitpaint']:
        cic.paint(pos, mesh, 1.0, transform=transform, mode='ignore', period=100)
    with t['oldpaint']:
        cic.paint_old(pos, mesh, 1.0, transform=transform, mode='ignore', period=100)

    with t['jitreadout']:
        cic.readout(mesh, pos, transform=transform, mode='ignore', period=100)
    with t['oldreadout']:
        cic.readout_old(mesh, pos, transform=transform, mode='ignore', period=100)
#test_speed()
