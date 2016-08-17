import numpy
import pfft
import mpsort
from . import domain
from . import window
from mpi4py import MPI

class slabiter(object):
    def __init__(self, field):
        # we iterate over the slowest axis to gain locality.
        axissort = numpy.argsort(field.strides)[::-1]
        axis = axissort[0]

        self.optimized_view = field.transpose(axissort)
        self.nslabs = field.shape[axis]

        optx = [xx.transpose(axissort) for xx in field.x]
        opti = [ii.transpose(axissort) for ii in field.i]
        self.x = xslabiter(axis, self.nslabs, optx)
        self.i = xslabiter(axis, self.nslabs, opti)

    def __iter__(self):
        for irow in range(self.nslabs):
            yield self.optimized_view[irow]

class xslabiter(slabiter):
    def __init__(self, axis, nslabs, optx):
        self.axis = axis
        self.nslabs = nslabs
        self.optx = optx

    def __iter__(self):
        for irow in range(self.nslabs):
            kk = [x[0] if d != self.axis else x[irow] for d, x in enumerate(self.optx)]
            yield kk


class Field(numpy.ndarray):
    """ Base class for RealField and ComplexField.

        It only supports those two subclasses.
    """
    def copy(self):
        other = self.__class__(self.pm)
        other[...] = self
        return other

    def add_attrs(self, buffer, pm):
        """ Used internally to add shortcuts of attributes from pm """
        self.pm = pm
        self.partition = pm.partition
        self.BoxSize = pm.BoxSize
        self.Nmesh = pm.Nmesh
        if isinstance(self, RealField):
            self.start = self.partition.local_i_start
            self.global_shape = pm.Nmesh
            self.x = pm.x
            self.i = pm.i_ind
        else:
            self.start = self.partition.local_o_start
            self.global_shape = pm.Nmesh.copy()
            self.global_shape[-1] = self.global_shape[-1] // 2 + 1
            self.x = pm.k
            self.i = pm.o_ind

        self.slices = tuple([
                slice(s, s + n)
                for s, n in zip(self.start, self.shape)
                ])

        self.slabs = slabiter(self)

    def sort(self, out=None):
        """ Sort the field to 'C'-order, partitioned by MPI ranks. Save the
            result to flatiter.

            Parameters
            ----------
            out : numpy.flatiter
                A flatiter to store the 'C' order. If not a flatiter, the .flat
                attribute is used.

            Returns
            -------
            numpy.flatiter : the flatiter provided or created.

            Notes
            -----
            Set `flatiter` to self for an 'inplace' sort.
        """
        ind = numpy.ravel_multi_index(numpy.mgrid[self.slices], self.global_shape)

        if out is None:
            out = numpy.empty_like(self)

        if not isinstance(out, numpy.flatiter):
            out = out.flat

        assert isinstance(out, numpy.flatiter)
        assert len(out) == self.size

        return mpsort.sort(self.flat, orderby=ind.flat, comm=self.pm.comm, out=out)

    def unsort(self, flatiter):
        """ Unsort c-ordered field values to the field.

            Parameters
            ----------
            flatiter : numpy.flatiter

            Notes
            -----
            self is updated. `array` does not have to be C_CONTIGUOUS flat iterator of array is used.
        """
        if not isinstance(flatiter, numpy.flatiter):
            flatiter = flatiter.flat

        assert isinstance(flatiter, numpy.flatiter)
        assert len(flatiter) == self.size

        ind = numpy.ravel_multi_index(numpy.mgrid[self.slices], self.global_shape)
        mpsort.permute(flatiter, argindex=ind.flat, comm=self.pm.comm, out=self.flat)

    def resample(self, out):
        """ Resample the Field by filling 0 or truncating modes.
            Convert from and between Real/Complex automatically.

            Parameters
            ----------
            out : Field
                must be provided because it is a different PM. Can be RealField or ComplexField

        """
        assert isinstance(out, Field)

        if all(out.Nmesh == self.Nmesh):
            # no resampling needed. Just do Fourier transforms.
            if isinstance(self, RealField) and isinstance(out, ComplexField):
                self.r2c(out)
            if isinstance(self, RealField) and isinstance(out, RealField):
                out[...] = self
            if isinstance(self, ComplexField) and isinstance(out, RealField):
                self.c2r(out)
            if isinstance(self, ComplexField) and isinstance(out, ComplexField):
                out[...] = self
            return out

        if isinstance(self, RealField):
            self = self.r2c()

        if isinstance(out, RealField):
            complex = ComplexField(out.pm)
        else:
            complex = out

        complex[...] = 0.0

        tmp = numpy.empty_like(self)

        self.sort(out=tmp)

        # indtable stores the index in pmsrc for the mode in pmdest
        # since pmdest < pmsrc, all items are alright.
        indtable = [reindex(self.Nmesh[d], out.Nmesh[d]) for d in range(self.ndim)]

        ind = build_index(
                [t[numpy.r_[s]]
                for t, s in zip(indtable, complex.slices) ],
                self.global_shape)

        # fill the points that has values in pmsrc
        mask = ind >= 0
        # their indices
        argind = ind[mask]
        # take the data

        data = mpsort.take(tmp.flat, argind, self.pm.comm)

        # fill in the value
        complex[mask] = data

        if isinstance(out, RealField):
            complex.c2r(out)

        return out

class RealField(Field):
    def __new__(kls, pm):
        buffer = pfft.LocalBuffer(pm.partition)
        self = buffer.view_input(type=kls)
        Field.add_attrs(self, buffer, pm)
        return self

    def r2c(self, out=None):
        """ 
        Perform real to complex FFT on the internal canvas.

        The complex field will be dimensionless; this is to ensure if NormalizeDC
        is applyed, c2r produces :math:`1 + \delta` as expected.

        (To obtain CFT, multiply by :math:`L^3` from the :math:`dx^3` factor )

        Therefore, the zeroth component of the complex field is :math:`\\bar\\rho`.

        """
        if out is None:
            out = ComplexField(self.pm)

        assert isinstance(out, ComplexField)

        self.pm.forward.execute(self.base, out.base)

        # PFFT normalization, same as FastPM
        out[...] *= numpy.prod(self.pm.Nmesh ** -1.0)
        return out


    def paint(self, pos, mass=1.0, method="cic", transform=None, hold=False):
        """ 
        Paint particles into the internal real canvas. 

        Transform the particle field given by pos and mass
        to the overdensity field in fourier space and save
        it in the internal storage. 
        A multi-linear CIC approximation scheme is used.

        The function can be called multiple times: 
        the result is cummulative. In a multi-step simulation where
        :py:class:`ParticleMesh` object is reused,  before calling 
        :py:meth:`paint`, make sure the canvas is cleared with :py:meth:`clear`.

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation unit

        mass   : scalar or array_like (,)
            mass of particles in simulation unit

        hold   : bool
            If true, do not clear the current value in the field.

        Notes
        -----
        the painter operation conserves the total mass. It is not the density.

        """
        # Transform from simulation unit to local grid unit.
        if not transform:
            transform = self.pm.affine

        if method in window.methods:
            method = window.methods[method]

        if not hold:
            self[...] = 0

        method.paint(self, pos, mass, transform=transform)

    def readout(self, pos, out=None, method="cic", transform=None):
        """ 
        Read out from real field at positions

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation  unit

        Returns
        -------
        rt     : array_like (,)
            read out values from the real field.

        """
        if not transform:
            transform = self.pm.affine

        method = window.methods[method]

        return method.readout(self, pos, out=out, transform=transform)


class ComplexField(Field):
    def __new__(kls, pm):
        buffer = pfft.LocalBuffer(pm.partition)
        self = buffer.view_output(type=kls)
        Field.add_attrs(self, buffer, pm)
        return self

    def c2r(self, out=None):
        if out is None:
            out = RealField(self.pm)
        assert isinstance(out, RealField)
        self.pm.backward.execute(self.base, out.base)
        return out

def build_index(indices, fullshape):
    """
        Build a linear index array based on indices on an array of fullshape.
        This is similar to numpy.ravel_multi_index.

        index value of -1 will on any axes will be translated to -1 in the final.

        Parameters:
            indices : a tuple of index per dimension.

            fullshape : a tuple of the shape of the full array

        Returns:
            ind : a 3-d array of the indices of the coordinates in indices in
                an array of size fullshape. -1 if any indices is -1.

    """
    localshape = [ len(i) for i in indices]
    ndim = len(localshape)
    ind = numpy.zeros(localshape, dtype='i8')
    for d in range(len(indices)):
        i = indices[d]
        i = i.reshape([-1 if dd == d else 1 for dd in range(ndim)])
        ind[...] *= fullshape[d]
        ind[...] += i

    mask = numpy.zeros(localshape, dtype='?')

    # now mask out bad points by -1
    for d in range(len(indices)):
        i = indices[d]
        i = i.reshape([-1 if dd == d else 1 for dd in range(ndim)])
        mask |= i == -1

    ind[mask] = -1
    return ind

def reindex(Nsrc, Ndest):
    """ returns the index in the frequency array for corresponding
        k in Nsrc and composes Ndest

        For those Ndest that doesn't exist in Nsrc, return -1

        Example:
        >>> reindex(8, 4)
        >>> array([0, 1, 2, 7])
        >>> reindex(4, 8)
        >>> array([ 0,  1,  2, -1, -1, -1,  -1,  3])

    """
    reindex = numpy.arange(Ndest)
    reindex[Ndest // 2 + 1:] = numpy.arange(Nsrc - Ndest // 2 + 1, Nsrc, 1)
    reindex[Nsrc // 2 + 1: Ndest -Nsrc //2 + 1] = -1
    return reindex

class ParticleMesh(object):
    """
    ParticleMesh provides an interface to solver for forces
    with particle mesh method

    ParticleMesh does not deal with memory. Use RealField(pm) and ComplexField(pm)
    to create memory buffers.

    Attributes
    ----------
    np      : array_like (npx, npy)
        The shape of the process mesh. This is the number of domains per direction.
        The product of the items shall equal to the size of communicator. 
        For example, for 64 rank job, np = (8, 8) is a good choice.
        Since for now only 3d simulations are supported, np must be of length-2.
        The default is try to split the total number of ranks equally. (eg, for
        a 64 rank job, default is (8, 8)

    comm    : :py:class:`MPI.Comm`
        the MPI communicator, (default is MPI.COMM_WORLD)

    Nmesh   : array of int
        number of mesh points per side.

    BoxSize : float
        size of box

    domain   : :py:class:`pmesh.domain.GridND`
        domain decomposition (private)

    partition : :py:class:`pfft.Partition`
        domain partition (private)

    w   : list
        a list of the circular frequencies along each direction (-pi to pi)
    k   : list
        a list of the wave numbers k along each direction (- pi N/ L to pi N/ L)
    x   : list
        a list of the position along each direction (-L/2 to L/ 2). x is conjugate of k.
    r   : list
        a list of the mesh position along each direction (-N/2 to N/2). r is conjugate of w.

    """

    def __init__(self, Nmesh, BoxSize=1.0, comm=None, np=None, dtype='f8', plan_method='estimate'):
        """ create a PM object.  """
        if comm is None:
            comm = MPI.COMM_WORLD

        self.comm = comm

        if np is None:
            if len(Nmesh) >= 3:
                np = pfft.split_size_2d(self.comm.size)
            else:
                np = [self.comm.size]

        dtype = numpy.dtype(dtype)
        if dtype is numpy.dtype('f8'):
            forward = pfft.Type.PFFT_R2C
            backward = pfft.Type.PFFT_C2R
        elif dtype is numpy.dtype('f4'):
            forward = pfft.Type.PFFTF_R2C
            backward = pfft.Type.PFFTF_C2R
        else:
            raise ValueError("dtype must be f8 or f4")

        self.procmesh = pfft.ProcMesh(np, comm=comm)
        self.Nmesh = numpy.array(Nmesh, dtype='i8')
        self.BoxSize = numpy.empty(len(Nmesh), dtype='f8')
        self.BoxSize[:] = BoxSize
        self.partition = pfft.Partition(forward,
            self.Nmesh,
            self.procmesh,
            pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_PADDED_R2C)

        bufferin = pfft.LocalBuffer(self.partition)
        bufferout = pfft.LocalBuffer(self.partition)

        plan_method = {
            "estimate": pfft.Flags.PFFT_ESTIMATE,
            "measure": pfft.Flags.PFFT_MEASURE,
            "exhaustive": pfft.Flags.PFFT_EXHAUSTIVE,
            } [plan_method]

        self.forward = pfft.Plan(self.partition, pfft.Direction.PFFT_FORWARD,
                bufferin, bufferout, forward,
                plan_method | pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_TUNE | pfft.Flags.PFFT_PADDED_R2C)
        self.backward = pfft.Plan(self.partition, pfft.Direction.PFFT_BACKWARD,
                bufferout, bufferin, backward, 
                plan_method | pfft.Flags.PFFT_TRANSPOSED_IN | pfft.Flags.PFFT_TUNE | pfft.Flags.PFFT_PADDED_C2R)

        self.domain = domain.GridND(self.partition.i_edges, comm=self.comm)

        k = []
        x = []
        w = []
        r = []
        o_ind = []
        i_ind = []

        for d in range(self.partition.ndim):
            t = numpy.ones(self.partition.ndim, dtype='intp')
            s = numpy.ones(self.partition.ndim, dtype='intp')
            t[d] = self.partition.local_i_shape[d]
            s[d] = self.partition.local_o_shape[d]

            i_indi = numpy.arange(t[d], dtype='intp') + self.partition.local_i_start[d]
            o_indi = numpy.arange(s[d], dtype='intp') + self.partition.local_o_start[d]

            wi = numpy.arange(s[d], dtype='f4') + self.partition.local_o_start[d] 
            ri = numpy.arange(t[d], dtype='f4') + self.partition.local_i_start[d] 

            wi[wi >= self.Nmesh[d] // 2] -= self.Nmesh[d]
            ri[ri >= self.Nmesh[d] // 2] -= self.Nmesh[d]

            wi *= (2 * numpy.pi / self.Nmesh[d])
            ki = wi * self.Nmesh[d] / self.BoxSize[d]
            xi = ri * self.BoxSize[d] / self.Nmesh[d]

            o_ind.append(o_indi.reshape(s))
            i_ind.append(i_indi.reshape(t))
            w.append(wi.reshape(s))
            r.append(ri.reshape(t))
            k.append(ki.reshape(s))
            x.append(xi.reshape(t))

        self.i_ind = i_ind
        self.o_ind = o_ind
        self.w = w
        self.r = r
        self.k = k
        self.x = x

        # Transform from simulation unit to local grid unit.
        self.affine = window.Affine(self.partition.ndim,
                    translate=-self.partition.local_i_start,
                    scale=1.0 * self.Nmesh / self.BoxSize,
                    period = self.Nmesh)

    def decompose(self, pos, smoothing="cic"):
        """ 
        Create a domain decompose layout for particles at given
        coordinates.

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation  unit

        smoothing : float, string, or ResampleWindow
            if given as a string or ResampleWindow, use 0.5 * support.
            This is the size of the buffer region around a domain.

        Returns
        -------
        layout  : :py:class:domain.Layout
            layout that can be used to migrate particles and images
        to the correct MPI ranks that hosts the PM local mesh
        """
        if smoothing in window.methods:
            smoothing = window.methods[smoothing]
        if isinstance(smoothing, window.ResampleWindow):
            smoothing = smoothing.support * 0.5

        # Transform from simulation unit to global grid unit.
        def transform0(x):
            return self.affine.scale * x

        return self.domain.decompose(pos, smoothing=smoothing,
                transform=transform0)
