import warnings
warnings.warn("tsc.py is deprecated; switch to window.TSC", DeprecationWarning)
# numba version of the tsc.
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
    """ TSC approximation, painting points to Nmesh,
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

    if not mesh.flags['C_CONTIGUOUS']:
        raise ValueError('mesh must be C continguous')

    # scratch 
    if transform is None:
        transform = lambda x:x
    if period is None: period = 0
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
    Nmax = int(3 ** Ndim)
    ignore = False
    period = int(period)
    outbound = 0
    for i in range(Np):
        w = float(weights[i])
        for n in range(Nmax):
            ignore = False
            kernel = 1.0
            ind = 0
            for d in range(Ndim):
                intpos = numpy.rint(pos[i, d]) # NGP index
                diff = intpos - pos[i, d]
                rel = (n // 3**d) % 3 - 1 # maps offset (rel. to NGP) to (-1, 0, 1)
                targetpos = intpos + rel - 1
                if rel == -1: # before NGP
                    kernel *= 0.5 * (0.5+diff)*(0.5+diff)
                elif rel == 0: # NGP
                    kernel *= 0.75 - diff*diff
                else: # after NGPs
                    kernel *= 0.5 * (0.5-diff)*(0.5-diff)
                
                # wrap by period
                if period > 0:
                    while targetpos >= period:
                        targetpos -= period
                    while targetpos < 0:
                        targetpos += period
                if targetpos < 0 or \
                        targetpos >= mesh.shape[d]:
                    ignore = True
                    break
                ind += mesh.strides[d]*targetpos

            if ignore:
                outbound += 1
                continue
            meshflat[int(ind/mesh.itemsize)] += w * kernel

    return outbound

