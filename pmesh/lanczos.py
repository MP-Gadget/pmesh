"""
.. deprecated:: 0.1

"""
import warnings
warnings.warn("kernels are implemented in window.py", DeprecationWarning)
# painting with any kernel 
#
import numpy

def linear(dx, abs=numpy.abs):
    dx = abs(dx)
    result = (1.0 - dx)
    result[dx > 1] = 0
    return result
linear.support = 1
linear.integral = 1.0

def cubic(dx, abs=numpy.abs, alpha=-0.5):
    dx = abs(dx)
    mask1 = dx < 1.0
    mask2 = dx >= 1.0
    v1 = (alpha + 2) * dx ** 3 - (alpha + 3) * dx ** 2 + 1
    v2 = alpha * dx ** 3 - 5 * alpha * dx ** 2 + 8 * alpha * dx - 4 * alpha
    result = v1.copy()
    result[mask2]  = v2[mask2]
    result[dx > 2]  = 0
    return result

cubic.support = 2
cubic.integral = 1.0

def lanczos(a):
    sinc=numpy.sinc
    ainv = 1.0 / a
    def lanczos(dx):
        v = sinc(dx) * sinc(dx * ainv)
        v[dx > a] = 0
        v[dx < -a] = 0
        return v
    lanczos.support = int(numpy.ceil(a))
    dx = numpy.linspace(-a, a, 10000)
    lanczos.integral = numpy.trapz(lanczos(dx), dx)
    return lanczos

lanczos2 = lanczos(2)
lanczos3 = lanczos(3)

def kaiser(a, alpha):
    i0 = numpy.i0
    beta = numpy.pi * alpha
    def kaiser(dx):
        tmp = (1 - (dx / a) ** 2) ** 0.5
        v = i0(beta * tmp).reshape(dx.shape) / i0(beta)
        v[dx > a] = 0
        v[dx < -a] = 0
        return v
    kaiser.support = a
    dx = numpy.linspace(-a, a, 10000)
    kaiser.integral = numpy.trapz(kaiser(dx), dx)
    return kaiser

def paint(pos, mesh, weights=1.0, mode="raise", period=None, transform=None, window=linear):
    """ 

        Paint particles, painting points to Nmesh,
        each point has a weight given by weights.
        This does not give density.
        pos is supposed to be row vectors. aka for 3d input
        pos.shape is (?, 3).

        pos[:, i] should have been normalized in the range of [ 0,  mesh.shape[i] )

        thus z is the fast moving index

        mode can be :
            "raise" : raise exceptions if a particle is painted
             outside the mesh
            "ignore": ignore particle contribution outside of the mesh
        period can be a scalar or of length len(mesh.shape). if period is given
        the particles are wrapped by the period.

        transform is a function that transforms pos to mesh units:
        transform(pos[:, 3]) -> meshpos[:, 3]
    """
    pos = numpy.array(pos)
    chunksize = 1024 * 16 * 4
    Ndim = pos.shape[-1]
    Np = pos.shape[0]

    if transform is None:
        transform = lambda x:x

    if not hasattr(window, 'support'):
        raise ValueError("Window function must declear its support (per side) as an attribute, e.g. bilinear.support = 1.")

    support = window.support

    neighbours = numpy.arange((2 * support) ** Ndim)[:, None]
    neighbours = neighbours // (2 * support) ** numpy.arange(Ndim)[None, :]
    neighbours %= 2 * support
    neighbours -= (support - 1)

    for start in range(0, Np, chunksize):
        chunk = slice(start, start+chunksize)
        if numpy.isscalar(weights):
          wchunk = weights
        else:
          wchunk = weights[chunk]
        if mode == 'raise':
            gridpos = transform(pos[chunk])
            rmi_mode = 'raise'
            intpos = numpy.intp(numpy.floor(gridpos))
        elif mode == 'ignore':
            gridpos = transform(pos[chunk])
            rmi_mode = 'raise'
            intpos = numpy.intp(numpy.floor(gridpos))

        for i, neighbour in enumerate(neighbours):
            neighbour = neighbour[None, :]
            targetpos = intpos + neighbour
            kernel = window(gridpos - targetpos).prod(axis=-1)
            #dx = gridpos - targetpos
            #kernel = window((dx ** 2).sum(axis=-1) ** 0.5)
            add = wchunk * (kernel / window.integral)

            if period is not None:
                numpy.remainder(targetpos, period, targetpos)

            if mode == 'ignore':
                # filter out those outside of the mesh
                mask = (targetpos >= 0).all(axis=-1)
                for d in range(Ndim):
                    mask &= (targetpos[..., d] < mesh.shape[d])
                targetpos = targetpos[mask]
                add = add[mask]

            if len(targetpos) > 0:
                targetindex = numpy.ravel_multi_index(
                        targetpos.T, mesh.shape, mode=rmi_mode)
                u, label = numpy.unique(targetindex, return_inverse=True)
                mesh.flat[u] += numpy.bincount(label, add, minlength=len(u))

    return mesh

d = numpy.zeros((6, 6))
p = numpy.zeros((1, 2)) + 2.5

#paint(p, d)
#paint(p, d, window=bicubic, period=6)
#paint(p, d, window=lanczos2, period=6)
paint(p, d, window=lanczos3, period=6)
#paint(p, d, window=bilinear, period=6)
