import warnings
warnings.warn("numba based cic is deprecated; _window.pyx implementes a c version of the windows.", DeprecationWarning)
# numba version of the cic.
import numpy
import numba
import math
def paint(pos, mesh, weights=1.0, mode="raise", period=None, transform=None):
    if weights is None:
        weights = 1.0
    return driver(pos, mesh, weights, mode, period, transform, paint_some)

def readout(mesh, pos, mode="raise", period=None, transform=None, out=None):
    if out is None:
        out = numpy.zeros(len(pos), 'f8')
    driver(pos, mesh, out, mode, period, transform, readout_some)
    return out

def driver(pos, mesh, weights, mode, period, transform,
        callback):
    """ CIC approximation (trilinear), painting points to Nmesh,
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
    chunksize = 1024 * 1024
    Ndim = pos.shape[-1]
    Np = pos.shape[0]

#    if not mesh.flags['C_CONTIGUOUS']:
#        raise ValueError('mesh must be C continguous')
#   I don't see why it must be C contiguous

    # scratch 
    if transform is None:
        transform = lambda x:x
    if period is None: period = numpy.zeros(Ndim, dtype='i4')
    else:
        p = numpy.empty(Ndim, dtype='i4')
        p[...] = period
        period = p

    for start in range(0, Np, chunksize):
        chunk = slice(start, start+chunksize)
        mypos = transform(pos[chunk])
        if numpy.isscalar(weights):
            wchunk = numpy.empty(chunksize)
            wchunk[:] = weights
        else:
            wchunk = weights[chunk]
        if callback(mypos, mesh, mesh.ravel(), wchunk, period) \
                and mode == "raise":
           raise ValueError("Some points are out of boundary")
    return mesh
@numba.jit(nopython=True)
def paint_some(pos, mesh, meshflat, weights, period):
    Ndim = pos.shape[1]
    Np = pos.shape[0]
    Nmax = int(2 ** Ndim)
    ignore = False
    outbound = 0
    for i in range(Np):
        w = float(weights[i])
        for n in range(Nmax):
            ignore = False
            kernel = 1.0
            ind = 0
            for d in range(Ndim):
                intpos = int(math.floor(pos[i, d]))
                diff = pos[i, d] - intpos
                rel = (n>>d) & 1
                targetpos = intpos + rel
                if rel > 0:
                    kernel *= diff
                else:
                    kernel *= (1.0 - diff)
                if period[d] > 0:
                    while targetpos >= period[d]:
                        targetpos -= period[d]
                    while targetpos < 0:
                        targetpos += period[d]
                if targetpos < 0 or \
                        targetpos >= mesh.shape[d]:
                    ignore = True
                    break
                ind = ind * mesh.shape[d] + targetpos
            #print targetpos, intpos, pos[i], kernel, w, ignore
            if ignore:
                outbound += 1
                continue
            meshflat[ind] += w * kernel

    return outbound



@numba.jit(nopython=True)
def readout_some(pos, mesh, meshflat, myvalue, period):
    Ndim = pos.shape[1]
    Np = pos.shape[0]
    Nmax = int(2 ** Ndim)
    ignore = False
    outbound = 0
    for i in range(Np):
        tmp = 0.0
        for n in range(Nmax):
            ignore = False
            kernel = 1.0
            ind = 0
            for d in range(Ndim):
                intpos = int(math.floor(pos[i, d]))
                diff = pos[i, d] - intpos
                rel = (n>>d) & 1
                targetpos = intpos + rel
                if rel > 0:
                    kernel *= diff
                else:
                    kernel *= (1.0 - diff)
                if period[d] > 0:
                    while targetpos >= period[d]:
                        targetpos -= period[d]
                    while targetpos < 0:
                        targetpos += period[d]
                if targetpos < 0 or \
                        targetpos >= mesh.shape[d]:
                    ignore = True
                    break
                ind = ind * mesh.shape[d] + targetpos
            #print i, n, targetpos, intpos, period, pos[i], mesh.shape, kernel, ignore
            if ignore:
                outbound += 1
                continue
            tmp += meshflat[ind] * kernel
        myvalue[i] = tmp
    return outbound
