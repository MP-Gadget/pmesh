from __future__ import absolute_import

import numpy
from abopt.vmad2 import ZERO, Engine, statement, programme, CodeSegment, Literal
from pmesh.pm import ParticleMesh, RealField, ComplexField

def create_grid(basepm, shift=0, dtype='f4'):
    """
        create uniform grid of particles, one per grid point on the basepm mesh
        this shall go to pmesh.
    """
    ndim = len(basepm.Nmesh)
    real = basepm.create('real')

    _shift = numpy.zeros(ndim, dtype)
    _shift[:] = shift
    # one particle per base mesh point
    source = numpy.zeros((real.size, ndim), dtype=dtype)

    for d in range(len(real.shape)):
        real[...] = 0
        for xi, slab in zip(real.slabs.i, real.slabs):
            slab[...] = (xi[d] + 1.0 * _shift[d]) * (real.BoxSize[d] / real.Nmesh[d])
        source[..., d] = real.value.flat
    return source

def nyquist_mask(factor, v):
    # any nyquist modes are set to 0 if the transfer function is complex
    mask = (numpy.imag(factor) == 0) | \
            ~numpy.bitwise_and.reduce([(ii == 0) | (ii == ni // 2) for ii, ni in zip(v.i, v.Nmesh)])
    return factor * mask

class ParticleMeshEngine(Engine):
    def __init__(self, pm, q=None):
        self.pm = pm
        if q is None:
            q = create_grid(pm, dtype='f4')
        self.q = q

    def get_x(self, s):
        x = numpy.float64(self.q)
        x[...] += s
        return x

    @statement(aout=['real'], ain=['complex'])
    def c2r(engine, real, complex):
        real[...] = complex.c2r()

    @c2r.defvjp
    def _(engine, _real, _complex):
        _complex[...] = _real.c2r_gradient()

    @statement(aout=['complex'], ain=['real'])
    def r2c(engine, complex, real):
        complex[...] = real.r2c()

    @r2c.defvjp
    def _(engine, _complex, _real):
        _real[...] = _complex.r2c_gradient()

    @statement(aout=['complex'], ain=['complex'])
    def decompress(engine, complex):
        return

    @decompress.defvjp
    def _(engine, _complex):
        _complex.decompress_gradient(out=Ellipsis)

    @staticmethod
    def _resample_filter(k, v, Neff):
        k0s = 2 * numpy.pi / v.BoxSize
        mask = numpy.bitwise_and.reduce([abs(ki) <= Neff//2 * k0 for ki, k0 in zip(k, k0s)])
        return v * mask

    @statement(aout=['mesh'], ain=['mesh'])
    def resample(engine, mesh, Neff):
        mesh.r2c(out=Ellipsis).apply(
            lambda k, v: engine._resample_filter(k, v, Neff),
            out=Ellipsis).c2r(out=Ellipsis)

    @resample.defvjp
    def _(engine, _mesh, Neff):
        _mesh.c2r_gradient().apply(
            lambda k, v: engine._resample_filter(k, v, Neff),
            out=Ellipsis).r2c_gradient(out=Ellipsis)

    @statement(aout=['layout'], ain=['s'])
    def decompose(engine, layout, s):
        x = engine.get_x(s)
        pm = engine.pm
        layout[...] = pm.decompose(x)

    @decompose.defvjp
    def _(engine, _layout, _s):
        _s[...] = ZERO

    @statement(aout=['mesh'], ain=['s', 'layout'])
    def paint(engine, s, mesh, layout):
        pm = engine.pm
        x = engine.get_x(s)
        mesh[...] = pm.create(mode='real')
        N = pm.comm.allreduce(len(x))
        mesh[...].paint(x, layout=layout, hold=False)
        # to have 1 + \delta on the mesh
        mesh[...][...] *= 1.0 * pm.Nmesh.prod() / N

    @paint.defvjp
    def _(engine, _s, _mesh, s, layout, _layout):
        pm = engine.pm
        _layout[...] = ZERO
        x = engine.get_x(s)
        N = pm.comm.allreduce(len(x))
        _s[...], junk = _mesh.paint_gradient(x, layout=layout, out_mass=False)
        _s[...][...] *= 1.0 * pm.Nmesh.prod() / N

    @statement(aout=['value'], ain=['s', 'mesh', 'layout'])
    def readout(engine, value, s, mesh, layout):
        pm = engine.pm
        x = engine.get_x(s)
        N = pm.comm.allreduce(len(x))
        value[...] = mesh.readout(x, layout=layout)

    @readout.defvjp
    def _(engine, _value, _s, _mesh, s, layout, mesh):
        pm = engine.pm
        x = engine.get_x(s)
        _mesh[...], _s[...] = mesh.readout_gradient(x, _value, layout=layout)

    @statement(aout=['complex'], ain=['complex'])
    def transfer(engine, complex, tf):
        complex.apply(lambda k, v: nyquist_mask(tf(k), v) * v, out=Ellipsis)

    @transfer.defvjp
    def _(engine, tf, _complex):
        _complex.apply(lambda k, v: nyquist_mask(numpy.conj(tf(k)), v) * v, out=Ellipsis)

    @statement(aout=['r'], ain=['field'])
    def norm(engine, field, r, metric=None):
        if isinstance(field, ComplexField):
            r[...] = field.cnorm(metric=metric)
        else:
            r[...] = field.cnorm()

    @norm.defvjp
    def _(engine, _field, _r, metric, field):
        if isinstance(field, ComplexField):
            _field[...] = field.cnorm_gradient(_r, metric=metric)
        else:
            _field[...] = field * (2 * _r)

    @statement(aout=['residual'], ain=['model'])
    def residual(engine, model, data, sigma, residual):
        d = model - data
        d[...] /= sigma
        residual[...] = d

    @residual.defvjp
    def _(engine, _model, _residual, data, sigma):
        g = _residual.copy()
        g[...] /= sigma
        _model[...] = g

    @statement(ain=['attribute', 'value'], aout=['attribute'])
    def assign_component(engine, attribute, value, dim):
        attribute[..., dim] = value

    @assign_component.defvjp
    def _(engine, _attribute, _value, dim):
        _value[...] = _attribute[..., dim]

    @statement(ain=['x'], aout=['y'])
    def assign(engine, x, y):
        y[...] = x.copy()

    @assign.defvjp
    def _(engine, _y, _x):
        _x[...] = _y

    @statement(ain=['x1', 'x2'], aout=['y'])
    def add(engine, x1, x2, y):
        y[...] = x1 + x2

    @add.defvjp
    def _(engine, _y, _x1, _x2):
        _x1[...] = _y
        _x2[...] = _y

    @statement(aout=['y'], ain=['x1', 'x2'])
    def multiply(engine, x1, x2, y):
        y[...] = x1 * x2

    @multiply.defvjp
    def _(engine, _x1, _x2, _y, x1, x2):
        _x1[...] = _y * x2
        _x2[...] = _y * x1

    @statement(ain=['x'], aout=['y'])
    def to_scalar(engine, x, y):
        y[...] = engine.pm.comm.allreduce((x[...] ** 2).sum(dtype='f8'))

    @to_scalar.defvjp
    def _(engine, _y, _x, x):
        _x[...] = x * (2 * _y)

def check_grad(code, yname, xname, init, eps, rtol, verbose=False):
    from numpy.testing import assert_allclose
    engine = code.engine
    comm = engine.pm.comm
    if isinstance(init[xname], numpy.ndarray) and init[xname].shape == engine.q.shape:
        cshape = engine.pm.comm.allreduce(engine.q.shape[0]), engine.q.shape[1]

        def cperturb(pos, ind, eps):
            pos = pos.copy()
            start = sum(comm.allgather(pos.shape[0])[:comm.rank])
            end = sum(comm.allgather(pos.shape[0])[:comm.rank + 1])
            if ind[0] >= start and ind[0] < end:
                old = pos[ind[0] - start, ind[1]]
                coord = pos[ind[0]-start].copy()
                pos[ind[0] - start, ind[1]] = old + eps
                new = pos[ind[0] - start, ind[1]]
            else:
                old, new, coord = 0, 0, 0
            diff = comm.allreduce(new - old)
            return pos

        def cget(pos, ind):
            if pos is ZERO: return 0
            start = sum(comm.allgather(pos.shape[0])[:comm.rank])
            end = sum(comm.allgather(pos.shape[0])[:comm.rank + 1])
            if ind[0] >= start and ind[0] < end:
                old = pos[ind[0] - start, ind[1]]
            else:
                old = 0
            return comm.allreduce(old)

    elif isinstance(init[xname], RealField):
        cshape = init[xname].cshape
        def cget(real, index):
            if real is ZERO: return 0
            return real.cgetitem(index)

        def cperturb(real, index, eps):
            old = real.cgetitem(index)
            r1 = real.copy()
            r1.csetitem(index, old + eps)
            return r1
    code = code.copy()
    code.to_scalar(x=yname, y='y')

    y, tape = code.compute('y', init=init, return_tape=True)
    gradient = tape.gradient()
    _x = gradient.compute('_' + xname, init={'_y' : 1.0})

    center = init[xname]
    init2 = init.copy()
    poor = []
    for index in numpy.ndindex(*cshape):
        x1 = cperturb(center, index, eps)
        x0 = cperturb(center, index, -eps)
        analytic = cget(_x, index)
        init2[xname] = x1
        y1 = code.compute('y', init2)
        init2[xname] = x0
        y0 = code.compute('y', init2)
        #logger.DEBUG("CHECKGRAD: %s" % (y1, y0, y1 - y0, get_pos(code.engine, _x, index) * 2 * eps))
        if verbose:
            print(index, (x1 - x0)[...].max(), y1, y0, y, y1 - y0, cget(_x, index) * 2 * eps)

        if not numpy.allclose(y1 - y0, cget(_x, index) * 2 * eps, rtol=rtol):
            poor.append([index, y1 - y0, cget(_x, index) * 2 * eps])

    if len(poor) != 0:
        print('\n'.join(['%s' % a for a in poor]))
        raise AssertionError("gradient of %d / %d parameters are bad." % (len(poor), numpy.prod(cshape)))

