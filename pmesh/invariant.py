from . import _invariant
import numpy

def get_index(x, Nmesh, compressed=True, maxlength=None):
    """
    Return the scale invariant index.

    Parameters
    ----------
    x : array_like (..., d)
        integer index of points. It shall range from
        - Nmesh // 2 to Nmesh // 2 - 1.
    Nmesh : array_like
        size of the mesh, broadcast to (d,)
    compressed : bool
        if the last axis is compressed. Skip the
        non-positive half in the index if True.
    maxlength : int or None
        if given, set return to -1 if the index is
        greater or equal to maxlength. This hits
        a faster code path.

    Returns
    -------
    ind : array_like (...)
        the scale invariant index of the queried
        points. It is guarentted that if a point
        is closer to zero in the Linf distance, then
        the index is smaller. Therefore,
        one can select the long wave modes by keeping
        the points with smaller indices.
        -1 if the mode is outside the Nmesh range.
    """

    assert numpy.ndim(x) >= 2
    ndim = numpy.shape(x)[-1]
    Nmesh = numpy.broadcast_to(Nmesh, ndim).astype('intp')
    oldshape = numpy.shape(x)[:-1]
    r = _invariant.get_index(x.reshape(-1, ndim), Nmesh, compressed, maxlength=maxlength)
    r = r.reshape(oldshape)
    return r
