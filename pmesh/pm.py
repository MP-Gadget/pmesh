import functools
import operator

import numpy
import pfft
import mpsort
from . import domain
from .window import FindResampler, Affine

import warnings

try:
    from numpy.lib.mixins import NDArrayOperatorsMixin as NDArrayLike
except ImportError:
    warnings.warn('numpy version is low. Update to > 1.13.0. Falling back to older version of operator overides.')
    class NDArrayLike(object):
        __array_priority__ = 20.
        def __radd__(self, other): return self.__add__(other)
        def __rmul__(self, other): return self.__mul__(other)

        def __eq__(self, other):
            return self[...] == other

        def __add__(self, other):
            r = self.copy()
            r[...] += other
            return r

        def __sub__(self, other):
            r = self.copy()
            r[...] -= other
            return r

        def __rsub__(self, other):
            r = self.copy()
            r[...] *= -1
            r[...] += other
            return r

        def __mul__(self, other):
            r = self.copy()
            r[...] *= other
            return r

        def __div__(self, other):
            r = self.copy()
            r[...] /= other
            return r

        __truediv__ = __div__
        def __rdiv__(self, other):
            r = self.copy()
            r[...] = other / self[...]
            return r
        __rtruediv__ = __rdiv__

        def __abs__(self):
            r = self.copy()
            r[...] = abs(self[...])
            return r

        def __pow__(self, other):
            r = self.copy()
            r[...] = self[...] ** other
            return r

        def __neg__(self):
            r = self.copy()
            r[...] = -self[...]
            return r

from mpi4py import MPI

import numbers # for testing Numbers
import warnings
import functools
from collections import OrderedDict

_gettype = type

def is_inplace(out):
    return out is Ellipsis

class slab(numpy.ndarray):
    pass

class slabiter(object):
    def __init__(self, field, value):
        # we iterate over the slowest axis to gain locality.
        if field.ndim == 2:
            axis = 2
            self.optimized_view = value[None, ...]
            self.nslabs = 1
            self.optx = [xx[None, ...] for xx in field.x]
            self.opti = [ii[None, ...] for ii in field.i]
        else:
            axissort = numpy.argsort(field.value.strides)[::-1]
            axis = axissort[0]

            self.optimized_view = value.transpose(axissort)
            self.nslabs = field.shape[axis]

            self.optx = [xx.transpose(axissort) for xx in field.x]
            self.opti = [ii.transpose(axissort) for ii in field.i]
        self.axis = axis
        self.Nmesh = field.Nmesh
        self.BoxSize = field.BoxSize
        self.x = xslabiter(self, axis, self.nslabs, self.optx)
        self.i = xslabiter(self, axis, self.nslabs, self.opti)

    def __iter__(self):
        for irow in range(self.nslabs):
            s = self.optimized_view[irow].view(type=slab)
            kk = [x[0] if d != self.axis else x[irow] for d, x in enumerate(self.optx)]
            ii = [x[0] if d != self.axis else x[irow] for d, x in enumerate(self.opti)]
            s.x = kk
            s.i = ii
            s.BoxSize = self.BoxSize
            s.Nmesh = self.Nmesh
            yield s

class xslab(list):
    def normp(self, p=2, zeromode=None):
        """ returns the p-norm of the vector, matching the broadcast shape.

            Parameters
            ----------
            p : float
                pnorm
            zeromode : float, or None
                set the zeromode to this value if not None.
        """
        kk = (sum([abs(ki) ** p for ki in self]))
        if zeromode is not None:
            kk[kk == 0] = zeromode
        return kk

class xslabiter(slabiter):
    """ iterating will yield the sparse coordinates of a list of slabs """
    def __init__(self, slabiter, axis, nslabs, optx):
        self.axis = axis
        self.BoxSize = slabiter.BoxSize
        self.Nmesh = slabiter.Nmesh
        self.nslabs = nslabs
        self.optx = optx

    def __iter__(self):
        for irow in range(self.nslabs):
            kk = [x[0] if d != self.axis else x[irow] for d, x in enumerate(self.optx)]
            slab = xslab(kk)
            slab.BoxSize = self.BoxSize
            slab.Nmesh = self.Nmesh
            yield slab


class Field(NDArrayLike):
    """ Base class for RealField and ComplexField.

        It only supports those two subclasses.
    """
    def __repr__(self):
        if hasattr(self, 'value'):
            return '%s:' % self.__class__.__name__ + repr(self.value)
        else:
            return '%s:' % self.__class__.__name__

    _HANDLED_TYPES = (numpy.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (Field,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, Field) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, Field) else x
                for x in out)

        result = getattr(ufunc, method)(*inputs, **kwargs)
        def cast(result):
            # booleans, cannot be reasonable Field objects
            # just return the ndarray
            if result.dtype == '?':
                return result
            # different shape, cannot be reasonable Field objects
            # just return the ndarray
            if result.shape != self.shape:
                return result
            # really only cast when we are using simple +-* **, etc.
            return self.pm.create(_gettype(self), value=result)
        if type(result) is tuple:
            # multiple return values
            return tuple(cast(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return cast(result)

    def _check_compatible(self, other):
        if isinstance(other, Field):
            if not isinstance(other, _gettype(self)):
                raise TypeError("type of two operands of cdot must be the same type")
        else:
            assert all(numpy.shape(other) == self.shape)

    def copy(self):
        return self.pm.create(_gettype(self), value=self.value)

    def __init__(self, pm, base=None):
        """ Used internally to add shortcuts of attributes from pm """

        partition = pm._get_partition(type(self))

        # create a new base object based on the given base object
        base = pfft.LocalBuffer(partition, base=base)

        self._base = base
        self.pm = pm
        self._partition = partition
        self.BoxSize = pm.BoxSize
        self.Nmesh = pm.Nmesh
        self.ndim = len(pm.Nmesh)

        if isinstance(self, RealField):
            self.value = base.view_input()
            self.start = partition.local_i_start
            self.cshape = numpy.array([e[-1] for e in partition.i_edges], dtype='intp')
        elif isinstance(self, (TransposedComplexField, UntransposedComplexField)):
            self.value = base.view_output()
            self.start = partition.local_o_start
            self.cshape = numpy.array([e[-1] for e in partition.o_edges], dtype='intp')
            self.real = self.value.real
            self.imag = self.value.imag
            self.plain = self.value.view(dtype=(self.real.dtype, 2))
        else:
            raise TypeError("Only RealField and ComplexField. No more subclassing");

        self.x = pm.create_coords(type(self), return_indices=False)
        self.i = pm.create_coords(type(self), return_indices=True)

        # copy over a few ndarray attributes
        self.flat = self.value.flat

        self.shape = self.value.shape
        self.size = self.value.size
        self.dtype = self.value.dtype

        # the slices in the full array
        self.slices = tuple([
                slice(s, s + n)
                for s, n in zip(self.start, self.shape)
                ])

        self.csize = functools.reduce(operator.mul, self.cshape, 1)

    def _ctol(self, index):
        oldindex = index
        index = numpy.array(index, copy=True)

        if len(index) == self.ndim + 1:
            value = self.plain
            index1 = index[:-1]
        elif len(index) == self.ndim:
            value = self.value
            index1 = index
        else:
            raise IndexError("Only vector index in global indexing is supported. for complex append 0 or 1 for real and imag")

        # negative indexing
        index1[index1 < 0] += self.Nmesh[index1 < 0]
        if all(index1 >= self.start) and all(index1 < self.start + self.shape):
            return value, tuple(list(index1 - self.start) + list(index[self.ndim:]))
        else:
            return value, None

    def cgetitem(self, index):
        """ get a value from absolute index collectively.
        """
        value, localindex = self._ctol(index)
        if localindex is not None:
            ret = value[localindex]
        else:
            ret = 0

        return self.pm.comm.allreduce(ret)

    def csetitem(self, index, y):
        """ get a value from absolute index collectively.
            maintains Hermitian conjugation.
            Returns the actually value that is set.

        """

        index = numpy.array(index, copy=True)
        value, localindex = self._ctol(index)
        if isinstance(self, BaseComplexField):
            dualindex = numpy.negative(index)
            if len(dualindex) == self.ndim + 1:
                dualindex[-1] *= -1

            dualindex[:self.ndim] += self.Nmesh
            dualindex[:self.ndim] %= self.Nmesh

            unused, duallocalindex = self._ctol(dualindex)
        else:
            # real field, no dual
            duallocalindex = None

        dualy = y
        if localindex is None:
            y = 0
        if duallocalindex is None:
            dualy = 0

        if len(index) == self.ndim + 1 and index[-1] == 1:
            dualy = -dualy
            if localindex is not None and duallocalindex is not None:
                if localindex == duallocalindex:
                    # self dual and imag
                    y = 0
                    dualy = 0
        elif len(index) == self.ndim:
            dualy = numpy.conjugate(dualy)
            if localindex is not None and duallocalindex is not None:
                if localindex == duallocalindex:
                    # self conjugate
                    dualy = dualy.real
                    y = y.real
        if localindex is not None:
            value[localindex] = y
        if duallocalindex is not None:
            value[duallocalindex] = dualy

        return self.pm.comm.allreduce(y)

    def __getitem__(self, index):
        return self.value.__getitem__(index)

    def __setitem__(self, index, value):
        return self.value.__setitem__(index, value)

    def __array__(self, dtype=None):
        return self.value

    @property
    def compressed(self):
        """
        Whether the field stored in compressed half space format.

        The compressed format only stores the non-negative plane of values along
        the last axis, per PFFT and FFTW convention. All operations implicitly
        assumes the field is hermitian.

        A RealField is never compressed.
        """
        if self.Nmesh[-1] == self.cshape[-1]:
            return False
        elif self.Nmesh[-1] // 2 + 1 == self.cshape[-1]:
            return True
        else:
            raise ValueError(
                "The 3d mesh shape (%s) and the complex field shape (%s) are inconsistent." % 
                (str(self.Nmesh), str(self.cshape))
)

    @property
    def slabs(self):
        return slabiter(self, self.value)

    def sort(self, out=None):
        warnings.warn("Use ravel instead of sort", DeprecationWarning, stacklevel=2)
        return self.ravel(out)

    def unsort(self, flatiter):
        warnings.warn("Use pm.unravel instead of unsort", DeprecationWarning, stacklevel=2)
        return self.unravel(flatiter)

    def ravel(self, out=None):
        """ Ravel the field to 'C'-order, partitioned by MPI ranks. Save the
            result to flatiter.

            Parameters
            ----------
            out : numpy.flatiter, or Ellipsis for inplace
                A flatiter to store the 'C' order. If not a flatiter, the .flat
                attribute is used.

            Returns
            -------
            numpy.flatiter : the flatiter provided or created.

            Notes
            -----
            Set `out` to or Ellisps self.value for an 'inplace' ravel.
        """
        if out is None:
            out = numpy.empty_like(self.value)

        if is_inplace(out):
            out = self.value

        if not isinstance(out, numpy.flatiter):
            out = out.flat

        assert isinstance(out, numpy.flatiter)
        assert len(out) == self.size
        if self.pm.comm.size > 1:
            ind = numpy.ravel_multi_index(numpy.mgrid[self.slices], self.cshape)
            return mpsort.sort(self.flat, orderby=ind.flat, comm=self.pm.comm, out=out)
        else:
            # optimize for a single rank -- directly copy the result
            out[...] = self.flat
            return out

    def unravel(self, flatiter):
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
        assert self.pm.comm.allreduce(len(flatiter)) == self.csize

        if self.pm.comm.size > 1:
            ind = numpy.ravel_multi_index(numpy.mgrid[self.slices], self.cshape)
            mpsort.permute(flatiter, argindex=ind.flat, comm=self.pm.comm, out=self.flat)
        else:
            # optimize for a single rank -- directly copy the result
            self.flat[...] = flatiter

    def cast(self, type, out=None):
        """ cast the field object to the given type, maintaining the meaning of the field.
        """

        type = _typestr_to_type(type)

        if out is None:
            out = self.pm.create(type=type)
        else:
            out = self.pm.create(type=type, base=out._base)

        assert isinstance(out, type)

        if isinstance(self, RealField) and isinstance(out, BaseComplexField):
            self.r2c(out)
        if isinstance(self, RealField) and isinstance(out, RealField):
            out.value[...] = self.value
        if isinstance(self, BaseComplexField) and isinstance(out, RealField):
            self.c2r(out)
        if isinstance(self, BaseComplexField) and isinstance(out, BaseComplexField):
            if _gettype(self) is not _gettype(out):
                tmp = self.pm.create(type=RealField, base=out._base)
                # do a c2r r2c to account for the transpose
                self.c2r(out=tmp).r2c(out=out)
            else:
                out.value[...] = self.value

        return out

    def resample(self, out):
        """ Resample the Field by filling 0 or truncating modes.
            Convert from and between Real/Complex automatically.

            Parameters
            ----------
            out : Field
                must be provided because it is a different PM. Can be RealField or (Tranposed/Untransposed)ComplexField

        """
        assert isinstance(out, Field)

        if all(out.Nmesh == self.Nmesh):
            # no resampling needed. Just do Fourier transforms.
            self.cast(type=_gettype(out), out=out)

        self = self.cast(type=TransposedComplexField)

        complex = out.pm.create(type=TransposedComplexField, base=out._base, value=0)

        tmp = numpy.empty_like(self.value)

        self.ravel(out=tmp)

        # indtable stores the index in pmsrc for the mode in pmdest
        # since pmdest < pmsrc, all items are alright.
        indtable = [reindex(self.Nmesh[d], out.Nmesh[d]) for d in range(self.value.ndim)]

        ind = build_index(
                [t[numpy.r_[s]]
                for t, s in zip(indtable, complex.slices) ],
                self.cshape)

        # fill the points that has values in pmsrc
        mask = ind >= 0
        # their indices
        argind = ind[mask]
        # take the data

        data = mpsort.take(tmp.flat, argind, self.pm.comm)

        # fill in the value
        complex[mask] = data

        # ensure the down sample is real
        for i, slab in zip(complex.slabs.i, complex.slabs):
            mask = functools.reduce(numpy.bitwise_and,
                 [(n - ii) % n == ii
                    for ii, n in zip(i, complex.Nmesh)])
            slab.imag[mask] = 0

            # remove the nyquist of the output
            # FIXME: the nyquist is messy due to hermitian constraints
            # let's do not touch them till we know they are important.
            mask = functools.reduce(numpy.bitwise_or,
                 [ ii == n // 2
                   for ii, n in zip(i, complex.Nmesh)])
            slab[mask] = 0

            # also remove the nyquist of the input
            mask = functools.reduce(numpy.bitwise_or,
                 [ ii == n // 2
                   for ii, n in zip(i, self.Nmesh)])
            slab[mask] = 0

        if isinstance(out, RealField):
            complex.c2r(out)

        return out

    def preview(self, Nmesh=None, axes=None, resampler=None, method=None):
        """ gathers the mesh into as a numpy array, with
            (reduced resolution).

            The result is broadcast to all ranks, so this uses Nmesh.prod() per rank if all
            axes are preserved.

            Parameters
            ----------
            Nmesh : int, array_like, None
                The desired Nmesh of the result. Be aware this function
                allocates memory to hold A full Nmesh on each rank.
                None will not resample Nmesh.
            axes : list or None
                list of axes to preserve.

            method : string "upsample" or "downsample", or None
                upsample is like subsampling (faster) when Nmesh is lower resolution.
                if None, use upsample for upsampling (Nmesh >= self.Nmesh) and downsample for down sampling.

            Returns
            -------
            out : array_like
                An numpy array for the real density field.

        """
        if axes is None: axes = range(self.ndim)
        if not hasattr(axes, '__iter__'): axes = (axes,)
        else: axes = list(axes)

        if isinstance(self, BaseComplexField):
            self = self.c2r()

        if Nmesh is not None:
            # skip resampling if Nmesh is identical to current
            if all(Nmesh == self.Nmesh): Nmesh = None

        if Nmesh is not None:
            pm = self.pm.reshape(Nmesh)
            if method is None:
                if any(Nmesh < self.Nmesh): method = 'downsample'
                else : method = 'upsample'
            if method == 'downsample':
                out = pm.downsample(self, resampler=resampler, keep_mean=True)
            elif method == 'upsample':
                out = pm.upsample(self, resampler=resampler, keep_mean=True)
            else:
                raise ValueError("method can only be downsample or upsample")
        else:
            out = self

        result = numpy.zeros([out.cshape[i] for i in axes], dtype=out.dtype)
        local_slice = tuple([out.slices[i] for i in axes])

        # TODO: allow slicing along projected directions.
        out = out[...]

        if len(axes) != self.ndim:
            removeaxes = set(range(self.ndim)) - set(axes)
            all_axes = list(axes) + list(removeaxes)
            removeaxes = tuple(range(len(all_axes) - len(removeaxes), len(all_axes)))
            result[local_slice] += out.transpose(all_axes).sum(axis=removeaxes)
        else:
            result[local_slice] += out

        self.pm.comm.Allreduce(MPI.IN_PLACE, result)
        return result

    def apply(self, func, kind, out):
        """ implements all kinds of apply operations. see subclass members for documentation"""
        if out is None:
            out = self.pm.create(type=_gettype(self))

        if is_inplace(out):
            out = self

        if isinstance(out, numpy.ndarray):
            assert out.shape == self.value.shape
            outslabs = slabiter(self, out)
        else:
            assert isinstance(out, _gettype(self))
            assert out.value.shape == self.value.shape
            outslabs = slabiter(self, out.value)

        for x, i, islab, oslab in zip(self.slabs.x, self.slabs.i, self.slabs, outslabs):
            if kind == 'relative':
                oslab[...] = func(x, islab)
            elif kind == 'index':
                oslab[...] = func(i, islab)
            elif kind == 'absolute':
                oslab[...] = func(x, islab)
            elif kind == 'wavenumber':
                k = x
                oslab[...] = func(k, islab)
            elif kind == 'circular':
                w = [ ki * L / N for ki, L, N in zip(x, self.BoxSize, self.Nmesh)]
                oslab[...] = func(w, islab)
            else:
                raise ValueError("unknown kind of apply function.")
        return out


class RealField(Field):
    def __init__(self, pm, base=None):
        Field.__init__(self, pm, base)

    def r2c(self, out=None):
        """
        Perform real to complex transformation.

        """
        if out is None:
            out = TransposedComplexField(self.pm)

        if is_inplace(out):
            out = self

        if out is self:
            out = TransposedComplexField(self.pm, base=self._base)

        assert isinstance(out, (BaseComplexField,))

        if not self.pm._use_padded:
            # non-padded destroys input, so we fall back
            # to use the inplace transform
            # view out as self's type and copy the value
            self = self.pm.create(type=type(self), value=self.value, base=out._base)

        if self._base in out._base and out._base in self._base:
            # in place
            if isinstance(out, UntransposedComplexField):
                plan = self.pm.plans['ipforwardU']
            else:
                plan = self.pm.plans['ipforwardT']
        else:
            if isinstance(out, UntransposedComplexField):
                plan = self.pm.plans['forwardU']
            else:
                plan = self.pm.plans['forwardT']

        plan.execute(self._base, out._base)

        # PFFT normalization, same as FastPM
        out.value[...] *= numpy.prod(self.Nmesh ** -1.0)

        return out

    def ctranspose(self, axes):
        """ Collectively Transpose a RealField. This does not change the representation but actually
            replaces the coordinates according to the new set of axes.

            Notes
            -----
            This is currently implemented very inefficiently, with readout and paint operations.

        """
        # must be full rank axes.
        assert len(numpy.unique(axes)) == self.ndim
        assert numpy.max(axes) == self.ndim - 1

        # create a new pm with transposed BoxSize and Nmesh
        pm = self.pm.reshape(BoxSize=self.BoxSize[axes], Nmesh=self.Nmesh[axes])

        # for fancy indexing
        axes = numpy.array(axes, dtype='intp')

        q = self.pm.generate_uniform_particle_grid(shift=0)
        v = self.readout(q, resampler='nnb')

        # transpose the coordinates
        q = q[..., axes]

        layout = pm.decompose(q, smoothing='nnb')

        return pm.paint(q, mass=v, resampler='nnb', layout=layout)

    def csum(self, dtype=None):
        """ Collective mean. Sum of the entire mesh. (Must be called collectively)"""
        if dtype is None:
            dtype = self.dtype

        arg = numpy.argsort(self.value.strides)
        sum1 = self.value.transpose(arg[::-1])

        # first sum along the axis with the shortest strides
        # this would usually mean stabler results
        # when number of ranks are changed.
        for d in range(self.ndim):
            sum1 = sum1.sum(axis=-1, dtype=dtype)

        return self.pm.comm.allreduce(sum1)

    def cmean(self, dtype=None):
        """ Collective mean. Mean of the entire mesh. (Must be called collectively)"""
        return self.csum(dtype=dtype) / self.csize

    def readout(self, pos, hsml=None, out=None, resampler=None, transform=None, gradient=None, layout=None):
        """
        Read out from real field at positions

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation  unit
        hsml : array_like (, ndim)
            scaling of the resampling window per particle; or None for the kernel intrinsic size.
            (dimensionless)
        out : array_like (, ndim)
            output
        gradient : None or integer
            Direction to take the gradient of the window. The affine transformation
            is properly applied.
        resampler : None or string
            type of window, default to self.pm.resampler
        layout : Layout
            domain decomposition to use for the readout. The position is first
            routed to the target ranks and the result is reduced

        Returns
        -------
        rt     : array_like (,)
            read out values from the real field.

        """
        if not transform:
            transform = self.pm.affine

        if resampler is None:
            resampler = self.pm.resampler

        resampler = FindResampler(resampler)

        if layout is None:
            return resampler.readout(self.value, pos, hsml=hsml, out=out, transform=transform, diffdir=gradient)
        else:
            localpos = layout.exchange(pos)
            localhsml = exchange(layout, hsml)

            localresult = self.readout(localpos, hsml=localhsml, resampler=resampler,
                    transform=transform,
                    gradient=gradient,
                    out=None, layout=None)
            return layout.gather(localresult, out=out)

    def readout_vjp(self, pos, v, resampler=None, transform=None, gradient=None,
            out_self=None, out_pos=None, layout=None):
        """ back-propagate the gradient of readout.

            Returns a tuple of (out_self, out_pos), one of both can be False depending
            on the value of out_self and out_pos.

            Parameters
            ----------
            v: array
                current gradient over the result of readout.

            layout : Layout
                domain decomposition to use for the readout. The position is first
                routed to the target ranks and the result is reduced

            out_self: RealField, None, or False
                stored the backtraced gradient against self

                if False, then the gradient against self is not computed.
                if None, a new RealField is created and returned

            out_pos : array, None or False
                store the backtraced graident against pos

                if False, then the gradient against pos is not computed.
                if None, a new array is created and returned
        """
        if out_pos is not False:
            if gradient is not None:
                raise ValueError("gradient of gradient is not yet supported")
            if out_pos is None:
                out_pos = numpy.zeros_like(pos)
            if is_inplace(out_pos):
                out_pos = pos
            if out_pos is pos:
                # need to create a copy of pos because we use it later.
                pos = pos.copy()
            for d in range(pos.shape[1]):
                self.readout(pos, out=out_pos[:, d], resampler=resampler, transform=transform, gradient=d, layout=layout)
                out_pos[:, d] *= v

        if out_self is not False:
            if out_self is None:
                out_self = RealField(self.pm)
            if is_inplace(out_self):
                out_self = self

            # watch out: do this after using self, because out_self can be self.
            self.pm.paint(pos, mass=v, resampler=resampler, transform=transform, gradient=gradient, hold=False,
                    layout=layout, out=out_self)

        return out_self, out_pos


    def readout_jvp(self, pos, v_self=None, v_pos=None, resampler=None, transform=None, gradient=None, layout=None):
        """ f_i = W_qi A_q """
        jvp = numpy.zeros(len(pos))

        if v_pos is not None:
            for d in range(self.ndim):
                jvp[...] += self.readout(pos, resampler=resampler, transform=transform, gradient=d, layout=layout) * v_pos[..., d]

        if v_self is not None:
            jvp[...] += v_self.readout(pos, resampler=resampler, transform=transform, gradient=None, layout=layout)

        return jvp

    def paint(self, pos, mass=1.0, resampler=None, transform=None, hold=False, gradient=None, layout=None):
        warnings.warn("Use ParticleMesh.paint instead", DeprecationWarning, stacklevel=2)
        self.pm.paint(pos, mass=mass, resampler=resampler, transform=transform, hold=hold, gradient=gradient, layout=layout, out=self)

    def c2r_vjp(v, out=None):
        """ Back-propagate the gradient of c2r from self to out """
        out = v.r2c(out)
        # PFFT normalization, same as FastPM
        out.value[...] *= numpy.prod(out.pm.Nmesh ** 1.0)
        return out

    def apply(self, func, kind="relative", out=None):
        """ apply a function to the field.

            Parameters
            ----------
            func : callable
                func(r, y) where r is a list of r values that broadcasts into a full array.
                value of r depends on kind.

                y is the value of the field on the corresponding locations.

                `r.normp(p=2, zeromode=1)` would return `|r|^2` but set the zero mode (r == 0) to 1.

            kind : string
                The kind of value in r.
                'relative' means distance from `[-0.5 Boxsize, 0.5 BoxSize)`.
                'index' means `[0, Nmesh )`

            out : array_like or Field, or None.
                If provided, write into this object. Must be the same shape as self.

        """
        assert kind in ['relative', 'index', 'absolute']
        return Field.apply(self, func, kind, out)

    def cdot(self, other):
        self._check_compatible(other)
        return self.pm.comm.allreduce(numpy.sum(self[...] * other[...]))

    def cnorm(self):
        return self.cdot(self)

class BaseComplexField(Field):
    def __init__(self, pm, base=None):
        Field.__init__(self, pm, base)

    def _expand_hermitian(self, i, y):
        # is the field compressed?
        if not self.compressed:
            return y

        y = y.copy()
        # if a conjugate is not stored and not self, increase the weight
        # because we shall add it.
        mask = (i[-1] != 0) & (i[-1] != self.Nmesh[-1] // 2)
        y += mask * y
        return y

    def cnorm(self, metric=None, norm=lambda x: x.real **2 + x.imag**2):
        r"""compute the norm collectively. The conjugates are added too.

            This is effectively cdot(self).

            NORM = Self-conj + lower + upper

            .. math ::

                \sum_{m \in M} (self[m] * conjugate(other[m])
                            +   conjugate(self[m]) * other[m])
                             *  0.5  metric(k[m])
        """
        def filter2(k, y):
            y = norm(y)
            if metric is not None:
                k = k.normp(p=2) ** 0.5
                y *= metric(k)
            return y

        return self.pm.comm.allreduce(self\
                .apply(filter2)\
                .apply(self._expand_hermitian, kind='index', out=Ellipsis)\
                .value.sum())

    def cdot(self, other, metric=None):
        r""" Collective inner product between the independent modes of two Complex Fields.

            The real part of the result is effectively self.c2r().cdot(other.c2r()) / Nmesh.prod().

            FIXME: what does the imag part mean?

            Parameters
            ----------
            other : ComplexField
                the other field for the inner product
            metric: callable
                metric(k) gives the metric of each mode.

        """
        if isinstance(other, Field):
            if not isinstance(other, _gettype(self)):
                raise TypeError("type of two operands of cdot must be the same type")

        r = self.pm.create(type=_gettype(self), value=other)

        r.value[...] = numpy.conj(r.value[...])
        r.value[...] *= self.value

        r.apply(self._expand_hermitian, kind='index', out=Ellipsis)

        if metric is not None:
            r.apply(lambda k, y: y * metric(k.normp() ** 0.5), out=Ellipsis)

        return self.pm.comm.allreduce(r.value.sum())

    def cdot_vjp(self, v, metric=None):
        """ backtrace gradient of cdot against other. This is a partial gradient.
            This is currently only correct for cdot().real.
        """
        r = self * v

        if metric is not None:
            r.apply(lambda k, y: y * metric(k.normp() ** 0.5), out=Ellipsis)

        return r

    def c2r(self, out=None):
        if out is None:
            out = RealField(self.pm)
        if is_inplace(out):
            out = self

        if out is self:
            out = RealField(self.pm, self._base)

        assert isinstance(out, RealField)

        if not self.pm._use_padded:
            # non-padded destroys input, so we fall back
            # to using an inplace transform

            # view out as self, and copy the value
            self = self.pm.create(type=type(self), base=out._base, value=self.value)

        if out._base in self._base and self._base in out._base:
            # inplace
            if isinstance(self, UntransposedComplexField):
                plan = self.pm.plans['ipbackwardU']
            else:
                plan = self.pm.plans['ipbackwardT']
        else:
            if isinstance(self, UntransposedComplexField):
                plan = self.pm.plans['backwardU']
            else:
                plan = self.pm.plans['backwardT']

        plan.execute(self._base, out._base)

        return out

    def r2c_vjp(v, out=None):
        """ Back-propagate the gradient of r2c to self. """
        out = v.c2r(out)
        # PFFT normalization, same as FastPM
        out.value[...] *= numpy.prod(out.pm.Nmesh ** -1.0)
        return out

    def decompress_vjp(v, out=None):
        """ Back-propagate the gradient of decompress from self to out.
            If I change this mode in the .value array, how many modes are
            actually changed in order to maintain the hermitian?
        """
        if out is None:
            out = v.pm.create(type=_gettype(v))
        if is_inplace(out):
            out = v

        for i, a, b in zip(out.slabs.i, out.slabs, v.slabs):
            # modes that are self conjugates do not gain a factor
            mask = numpy.ones(a.shape, '?')
            for ii, n in zip(i, out.Nmesh):
               mask &= (n - ii) % n == ii
            a[~mask] = 2 * b[~mask]
            a[mask] = b[mask]
        return out

    def apply(self, func, kind="wavenumber", out=None):
        """ apply a function to the field, in-place.

            Parameters
            ----------
            func : callable
                func(k, y) where k is a list of k values that broadcasts into a full array.
                value of k depends on kind. y is the corrsponding value of field.

                y is the value of the field on the corresponding locations.

                `k.normp(p=2, zeromode=1)` would return `|k|^2` but set the zero mode (r == 0) to 1.

            kind : string
                The kind of value in k.
                'wavenumber' means wavenumber from [- 2 pi / L * N / 2, 2 pi / L * N / 2).
                'circular' means circular frequency from [- pi, pi).
                'index' means [0, Nmesh )

            out : array_like or Field, or None.
                If provided, write into this object. Must be the same shape as self.
        """
        assert kind in ['wavenumber', 'circular', 'index']
        return Field.apply(self, func, kind, out)

class UntransposedComplexField(BaseComplexField):
    """
        A complex field with untransposed representation. Faster for whitenoise,
        slower for r2c and c2r.
    """
    def __init__(self, pm, base=None):
        Field.__init__(self, pm, base)

class TransposedComplexField(BaseComplexField):
    """
        A complex field with transposed representation. Faster for r2c/c2r but slower for
        whitenoise
    """
    def __init__(self, pm, base=None):
        Field.__init__(self, pm, base)

# backward-compatbility, alias TranposedComplexField to ComplexField
ComplexField = TransposedComplexField

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

def exchange(layout, value):
    if value is None:
        return None

    if numpy.isscalar(value):
        value = numpy.array(value)

    if value.ndim != 0:
        localvalue = layout.exchange(value)
    else:
        localvalue = value
    return localvalue

def _typestr_to_type(typestr):

    if not isinstance(typestr, type):
        if typestr == 'real':
            typestr = RealField
        elif typestr == 'complex':
            typestr = ComplexField
        elif typestr == 'transposedcomplex':
            typestr = TransposedComplexField
        elif typestr == 'untransposedcomplex':
            typestr = UntransposedComplexField
        else:
            raise ValueError('mode must be real or complex, or ')

    if not issubclass(typestr, Field):
        raise TypeError("mode must be a subclass of %s" % str(Field))

    return typestr

def _init_i_coords(partition, Nmesh, BoxSize, dtype):
    x = []
    r = []
    i_ind = []

    for d in range(partition.ndim):
        t = numpy.ones(partition.ndim, dtype='intp')
        t[d] = partition.local_i_shape[d]

        i_indi = numpy.arange(t[d], dtype='intp') + partition.local_i_start[d]

        ri = numpy.arange(t[d], dtype=dtype) + partition.local_i_start[d]

        ri[ri >= Nmesh[d] // 2] -= Nmesh[d]

        xi = ri * BoxSize[d] / Nmesh[d]

        i_ind.append(i_indi.reshape(t))
        r.append(ri.reshape(t))
        x.append(xi.reshape(t))

    # FIXME: r
    return x, i_ind

def _init_o_coords(partition, Nmesh, BoxSize, dtype):
    k = []
    w = []
    o_ind = []

    for d in range(partition.ndim):
        s = numpy.ones(partition.ndim, dtype='intp')
        s[d] = partition.local_o_shape[d]

        o_indi = numpy.arange(s[d], dtype='intp') + partition.local_o_start[d]

        wi = numpy.arange(s[d], dtype=dtype) + partition.local_o_start[d]

        wi[wi >= Nmesh[d] // 2] -= Nmesh[d]

        wi *= (2 * numpy.pi / Nmesh[d])
        ki = wi * Nmesh[d] / BoxSize[d]

        o_ind.append(o_indi.reshape(s))
        w.append(wi.reshape(s))
        k.append(ki.reshape(s))

    # FIXME: w
    return k, o_ind


from weakref import WeakValueDictionary

_pm_cache = WeakValueDictionary()

class _pmtemplate(object):
    # subclass tuple to ensure ordered destruction.
    def __init__(self, procmesh, plans):
        self._tuple = (procmesh, plans)

    @property
    def procmesh(self):
        return self._tuple[0]
    @property
    def plans(self):
        return self._tuple[1]

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
        number of mesh points per side. The length decides the number of dimensions.

    dtype : dtype
        dtype of the buffers; if a complex dtype is given, the transforms will be c2c.
        the type of fields are still 'RealField' and 'ComplexField', though the RealField
        is actually made of complex numbers, and the ComplexField is no longer hermitian
        compressed.

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

    def __init__(self, Nmesh, BoxSize=1.0, comm=None, np=None, dtype='f8',
                    plan_method='estimate', resampler='cic'):
        """ create a PM object.  

            Parameters
            ----------
            plan_method : string
                method for planning, `estimate`, `exhaustive`, `measure`.
            resampler : string or ResampleWindow
                used to determine the default size of the domain decomposition
            np : the process mesh
                if None, automatically infer -- (n-1)d decomposition on (n)d mesh,
            Nmesh : tuple or alike
                size of the mesh. len(Nmesh) is the dimension of the system.

        """
        if comm is None:
            comm = MPI.COMM_WORLD

        self.comm = comm

        if len(Nmesh) == 1 and self.comm.size != 1:
            raise ValueError("Running 1d transforms on multiple ranks is not supported")

        if np is None:
            if len(Nmesh) >= 3:
                np = pfft.split_size_2d(self.comm.size)
            elif len(Nmesh) == 2:
                np = [self.comm.size]
            elif len(Nmesh) == 1:  # 1d transform, only 1 rank is supported. see above.
                np = []

        self.np = np

        if len(np) == len(Nmesh):
            # only implemented for non-padded and destroy input
            self._use_padded = False
            paddedflag = pfft.Flags.PFFT_DESTROY_INPUT
        else:
            self._use_padded = True
            paddedflag = pfft.Flags.PFFT_PRESERVE_INPUT | pfft.Flags.PFFT_PADDED_R2C | pfft.Flags.PFFT_PADDED_C2R

        dtype = numpy.dtype(dtype)

        if dtype == numpy.dtype('f8'):
            forward = pfft.Type.PFFT_R2C
            backward = pfft.Type.PFFT_C2R
        elif dtype == numpy.dtype('f4'):
            forward = pfft.Type.PFFTF_R2C
            backward = pfft.Type.PFFTF_C2R
        elif dtype == numpy.dtype('complex128'):
            forward = pfft.Type.PFFT_C2C
            backward = pfft.Type.PFFT_C2C
        elif dtype == numpy.dtype('complex64'):
            forward = pfft.Type.PFFTF_C2C
            backward = pfft.Type.PFFTF_C2C
        else:
            raise ValueError("dtype must be f8, f4, c16 or c8")

        self.Nmesh = numpy.array(Nmesh, dtype='i8')
        self.ndim = len(self.Nmesh)
        self.BoxSize = numpy.empty(len(Nmesh), dtype='f8')
        self.BoxSize[:] = BoxSize

        Nmesh = self.Nmesh
        BoxSize = self.BoxSize

        # if a similar ParticleMesh exists, use its
        # procmesh and plans,
        # this is to avoid creating too many MPI communicators,
        # which are a limited resource. (Intel has 16381, e.g.)
        # also see below where the instance self is inserted
        # to the weak dict.
        # the use of _addressof(comm) should be OK,
        # if we find a pm in the cache then the pm object
        # must have been holding a reference to the comm, so it
        # is alive.
        # if we don't find a pm then we'll create a new one anyways.

        _cache_args = (tuple(Nmesh), tuple(BoxSize),
                       MPI._addressof(comm), comm.rank, comm.size,
                       tuple(np), dtype, plan_method, paddedflag)

        template = _pm_cache.get(_cache_args, None)

        hastemplate = comm.allgather(template is not None)
        if not all(hastemplate):
            # some ranks the GC has already killed the cache; so we need to recreate
            # everything
            template = None

#        if comm.rank == 0:
#            print('hastemplate', hastemplate)
#            print(template, type(template), _cache_args)

        if template is not None:
            procmesh = template.procmesh
        else:
            procmesh = pfft.ProcMesh(np, comm=comm)

        plan_method = {
            "estimate": pfft.Flags.PFFT_ESTIMATE,
            "measure": pfft.Flags.PFFT_MEASURE,
            "exhaustive": pfft.Flags.PFFT_EXHAUSTIVE,
            } [plan_method]

        if template is not None:
            plans = template.plans
        else:
            plans = OrderedDict() # order dict implies ordered destruction.

            def make_duo(inplace, transposed):

                if transposed:
                    partition_flags = pfft.Flags.PFFT_TRANSPOSED_OUT | paddedflag
                    forward_flags = pfft.Flags.PFFT_TRANSPOSED_OUT | paddedflag
                    backward_flags = pfft.Flags.PFFT_TRANSPOSED_IN | paddedflag
                else:
                    partition_flags = paddedflag
                    forward_flags = paddedflag
                    backward_flags = paddedflag

                partition = pfft.Partition(forward,
                    Nmesh,
                    procmesh,
                    partition_flags)

                bufferin = pfft.LocalBuffer(partition)

                if not inplace:
                    bufferout = pfft.LocalBuffer(partition)
                else:
                    bufferout = bufferin

                fplan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD,
                    bufferin, bufferout, forward,
                    plan_method | forward_flags)
                bplan = pfft.Plan(partition, pfft.Direction.PFFT_BACKWARD,
                    bufferout, bufferin, backward,
                    plan_method  | backward_flags)
                return partition, fplan, bplan

            plans['partitionT'], plans['forwardT'], plans['backwardT'] = make_duo(False, True)
            junk, plans['ipforwardT'], plans['ipbackwardT'] = make_duo(True, True)

            plans['partitionU'], plans['forwardU'], plans['backwardU'] = make_duo(False, False)
            junk, plans['ipforwardU'], plans['ipbackwardU'] = make_duo(True, False)
            del junk

        # use the transpsoed partition for configuration space edges
        partition = plans['partitionT']

        self.domain = domain.GridND(partition.i_edges, comm=self.comm)

        self.procmesh = procmesh

        # Transform from simulation unit to local grid unit.
        self.affine = Affine(partition.ndim,
                    translate=-partition.local_i_start,
                    scale=1.0 * self.Nmesh / self.BoxSize,
                    period = self.Nmesh)

        # Transform from global grid unit to local grid unit.
        self.affine_grid = Affine(partition.ndim,
                    translate=-partition.local_i_start,
                    scale=1.0,
                    period = self.Nmesh)

        self.resampler = FindResampler(resampler)
        self.dtype = dtype
        self.plans = plans

        if template is None:
            template = _pmtemplate(procmesh, plans)

        self.template = template

        _pm_cache[_cache_args] = template

        self._coords = {}

    def _get_partition(self, field_type):
        if issubclass(field_type, RealField):
            # usually we use the transpsoed partition;
            # which is compatible for the real field either transposed
            # or not
            partition = self.plans['partitionT']
        elif issubclass(field_type, UntransposedComplexField):
            # for untransposed complex field.
            partition = self.plans['partitionU']
        elif issubclass(field_type, TransposedComplexField):
            partition = self.plans['partitionT']
        else:
            raise TypeError("not support type, internall Error")
        return partition

    def create_coords(self, field_type, return_indices=False):
        """ Create coordinate arrays. If return_indices is True, return
            the integer indices instead. The floating point coordiante
            arrays are of the same dtype as the ParticleMesh object,
            while the integral type coordiante arrays are of type intp.

            Returns
            -------
            x : (when return_indices is False) list of arrays, broadcastable to the right shape of the field;
                distance or wavenumber; between negative and positive.

            i : (when return_indices is True) list of arrays, integers (ranging from 0 to Nmesh)
        """
        field_type = _typestr_to_type(field_type)
        if field_type not in self._coords:
            partition = self._get_partition(field_type)
            if issubclass(field_type, RealField):
                self._coords[field_type] = _init_i_coords(partition, self.Nmesh, self.BoxSize, self.dtype)
            elif issubclass(field_type, BaseComplexField):
                self._coords[field_type] = _init_o_coords(partition, self.Nmesh, self.BoxSize, self.dtype)
            else:
                raise TypeError

        x, i = self._coords[field_type]
        if return_indices:
            return [ii.copy() for ii in i]
        return [xx.copy() for xx in x]

    @property
    def partition(self):
        return self.plans['partitionT']

    def resize(self, Nmesh):
        warnings.warn("ParticleMesh.resize method is deprecated. Use reshape method with full Nmesh as a tuple.", DeprecationWarning, stacklevel=2)
        return self.reshape(Nmesh=Nmesh)

    def reshape(self, Nmesh=None, BoxSize=None):
        """
            Create a reshaped ParticleMesh object, changing the resolution Nmesh, or even
            dimension.

            Parameters
            ----------
            Nmesh : int or array_like or None
                The new resolution

            Returns
            -------
            A ParticleMesh of the given resolution and transpose property
        """
        if Nmesh is None:
            Nmesh = self.Nmesh
        elif numpy.isscalar(Nmesh):
            Nmesh = [Nmesh for i in range(self.ndim)]

        if BoxSize is None:
            BoxSize = self.BoxSize[:len(Nmesh)]
        elif numpy.isscalar(BoxSize):
            BoxSize = [BoxSize for i in range(len(Nmesh))]

        if len(BoxSize) != len(Nmesh):
            raise ValueError("Dimension of BoxSize (%d) doesn't agree with Nmesh (%d); provide BoxSize explicitly." % (len(BoxSize), len(Nmesh)))

        return ParticleMesh(BoxSize=BoxSize,
                            Nmesh=Nmesh,
                            dtype=self.dtype,
                            comm=self.comm,
                            resampler=self.resampler,
                            np=self.np)

    def respawn(self, comm, np=None):
        """
            Create a new ParticleMesh object with the same geometry but on a new communicator.

            Notes
            -----
            Usually the communicator shall be a subcommunicator of self.comm, because otherwise
            there is no way to correctly make a barrier.

            Parameters
            ----------
            comm : MPI.Comm
                the new communicator
            np : list or int
                the process mesh topology

            Returns
            -------
            A new ParticleMesh on the given communicator;
        """
        return ParticleMesh(BoxSize=self.BoxSize,
                            Nmesh=self.Nmesh,
                            dtype=self.dtype,
                            comm=comm,
                            resampler=self.resampler,
                            np=np)

    def create(self, type=None, base=None, value=None, mode=None):
        """
            Create a field object.

            Parameters
            ----------
            type: string, or type
                'real', 'complex', 'untransposedcomplex',
                RealField, ComplexField, TransposedComplexField, UntransposedComplexField

            base : object, None
                Reusing the base attribute (physical memory) of an existing field
                object. Provide the attribute, not the field object. (`obj._base` not `obj`)

            value : array_like, None
                initialize the field with the values.
        """

        if mode is not None:
            warnings.warn("argument mode is deprecated. use type=%s instead" % mode, DeprecationWarning, stacklevel=2)

            if type is None:
                type = mode
            else:
                raise ValueError("both mode and type are specified, possiblity arguments are arranged in wrong order")

        type = _typestr_to_type(type)

        r = type(self, base=base)

        if value is not None:
            r[...] = value
        return r

    def unravel(self, type, flatiter):
        """ Unravel c-ordered field values.

            Parameters
            ----------
            type : type to unravel into, subclass of Field. (ComplexField, RealField, TransposedComplexField, UntransposedComplexField)
            flatiter : numpy.flatiter

            Returns
            -------
            r : RealField or ComplexField

            Notes
            -----
            `array` does not have to be C_CONTIGUOUS, as the flat iterator of array is used.
        """
        r = self.create(type=type)
        r.unravel(flatiter)
        return r

    def generate_whitenoise(self, seed, unitary=False, mean=0, type=TransposedComplexField, mode=None, base=None):
        """ Generate white noise to the field with the given seed.

            The scheme is supposed to be compatible with Gadget when the field is three-dimensional.

            Parameters
            ----------
            seed : int
                The random seed
            mean : float
                the mean of the field
            unitary : bool
                True to generate a unitary white noise where the amplitude is fixed to 1 and
                only the phase is random.
        """
        from .whitenoise import generate

        if mode is not None:
            warnings.warn("mode argument is deprecated, use type", DeprecationWarning, stacklevel=2)
            type = mode

        # first generate complex field
        type = _typestr_to_type(type)
        if type is RealField:
            complex_type = UntransposedComplexField
        else:
            complex_type = type

        complex = self.create(type=complex_type, base=base)
        generate(complex.value, complex.start, complex.Nmesh, seed, bool(unitary))

        # add mean
        def filter(k, v):
            mask = functools.reduce(numpy.bitwise_and, [ki == 0 for ki in k])
            v[mask] = mean
            return v

        complex.apply(filter, out=Ellipsis)

        # cast to the correct requested type
        return complex.cast(type=type, out=complex)

    def mesh_coordinates(self, dtype=None):
        partition = self.plans['partitionT']

        coord = numpy.indices(partition.local_i_shape, dtype).reshape(self.ndim, -1).T
        source = coord + partition.local_i_start
        return source

    def generate_uniform_particle_grid(self, shift=None, dtype=None, return_id=False):
        """
            create uniform grid of particles, one per grid point, in BoxSize coordinate.

            Parameters
            ----------
            shift : float, array_like, None
                shifting the grid by this much relative to the size of each grid cell.
                if array_like, per direction.

            dtype : dtype, or None
                dtype of the return value; default the same precision as the pm.

            return_id : boolean
                if True, return grid, id; id is the unique integer ID of this grid point.
                it is between 0 and total number of grid points (exclusive).

            Returns:
                grid : array_like (N, ndim)
                id   : array_like (N)
        """
        if dtype is None: dtype == self.dtype

        if shift is None:
            warnings.warn(
            "calling generate_uniform_particle_grid without a shift argument is deprecated."
            "use shift=0.5 for the previous default behavior. ", DeprecationWarning, 2)
            shift = 0.5

        shift = numpy.broadcast_to(shift, self.ndim)

        # one particle per base mesh point
        source = self.mesh_coordinates(dtype)

        source[...] += shift
        source[...] *= self.BoxSize / self.Nmesh
        source.flags.writeable = False

        if not return_id:
            return source

        isource = self.mesh_coordinates('i4')
        id = numpy.int64(isource[:, 0])
        for i in range(1, self.ndim):
            id[...] *= self.Nmesh[i]
            id[...] += isource[:, i]

        return source, id

    def decompose(self, pos, smoothing=None, transform=None):
        """
        Create a domain decompose layout for particles at given
        coordinates.

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation  unit

        smoothing : None, float, array_like, string, or ResampleWindow
            if given as a string or ResampleWindow, use 0.5 * support.
            This is the size of the buffer region around a domain.
            Default: None, use self.resampler

        Returns
        -------
        layout  : :py:class:domain.Layout
            layout that can be used to migrate particles and images
        to the correct MPI ranks that hosts the PM local mesh
        """
        if smoothing is None:
            smoothing = self.resampler

        try:
            smoothing = FindResampler(smoothing)
            smoothing = smoothing.support * 0.5
        except TypeError:
            pass

        if transform is None:
            transform = self.affine

        # Transform from simulation unit to global grid unit.
        def transform0(x):
            # shift is local per processor, thus do not use it.
            return transform.scale * x

        return self.domain.decompose(pos, smoothing=smoothing,
                transform=transform0)

    def paint(self, pos, hsml=None, mass=1.0, resampler=None, transform=None, hold=False, gradient=None, layout=None, out=None):
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

        hsml : array_like (, ndim)
            scaling of the resampling window per particle; or None for the kernel intrinsic size.
            (dimensionless)

        mass   : scalar or array_like (,)
            mass of particles in simulation unit

        hold   : bool
            If true, do not clear the current value in the field.

        gradient : None or integer
            Direction to take the gradient of the window. The affine transformation
            is properly applied.

        resampler: None or string
            type of window. Default : None, use self.pm.resampler

        layout : Layout
            domain decomposition to use for the readout. The position is first
            routed to the target ranks and the result is reduced

        Notes
        -----
        the painter operation conserves the total mass. It is not the density.

        """
        # Transform from simulation unit to local grid unit.
        if not transform:
            transform = self.affine

        if resampler is None:
            resampler = self.resampler

        resampler = FindResampler(resampler)

        if out is None:
            out = self.create(type=RealField)

        if not hold:
            out.value[...] = 0

        if layout is None:
            resampler.paint(out.value, pos, hsml=hsml, mass=mass, transform=transform, diffdir=gradient)
            return out
        else:
            localpos = layout.exchange(pos)
            localmass = exchange(layout, mass)
            localhsml = exchange(layout, hsml)

            return self.paint(localpos, mass=localmass,
                    hsml=localhsml,
                    resampler=resampler,
                    transform=transform,
                    hold=hold,
                    gradient=gradient,
                    layout=None, out=out)


    def paint_jvp(self, pos, mass=1.0, v_pos=None, v_mass=None, resampler=None, transform=None, gradient=None, layout=None, out=None):
        """ A_q = W_qi M_i """
        assert gradient is None # second order is not supported yet

        if out is None:
            out = self.create(type=RealField)

        out[...] = 0
        if v_pos is not None:
            for d in range(pos.shape[1]):
                self.paint(pos, mass=v_pos[..., d] * mass,
                    resampler=resampler, transform=transform, gradient=d, hold=True, layout=layout, out=out)

        if v_mass is not None:
            self.paint(pos, mass=v_mass,
                resampler=resampler, transform=transform, gradient=None, hold=True, layout=layout, out=out)
        return out

    def paint_vjp(self, v, pos, mass=1.0, resampler=None, transform=None, gradient=None,
            out_pos=None, out_mass=None, layout=None):
        """ back-propagate the gradient of paint from v.

            Parameters
            ----------
            layout : Layout
                domain decomposition to use for the readout. The position is first
                routed to the target ranks and the result is reduced

            out_mass: array , None, or False
                stored the backtraced gradient against mass

                if False, then the gradient against mass is not computed.
                if None, a new RealField is created and returned

            out_pos : array, None or False
                store the backtraced graident against pos

                if False, then the gradient against pos is not computed.
                if None, a new array is created and returned

        """
        if out_pos is not False:
            if gradient is not None:
                raise ValueError("gradient of gradient is not yet supported")
            if out_pos is None:
                out_pos = numpy.zeros_like(pos)
            if is_inplace(out_pos):
                out_pos = pos

            if out_pos is pos:
                pos = pos.copy()

            for d in range(pos.shape[1]):
                v.readout(pos, out=out_pos[:, d], resampler=resampler, transform=transform, gradient=d, layout=layout)
                out_pos[..., d] *= mass

        if out_mass is not False:
            if out_mass is None:
                out_mass = numpy.zeros(len(pos))
            if is_inplace(out_mass):
                out_mass = mass
            v.readout(pos, out=out_mass, resampler=resampler, transform=transform, gradient=gradient, layout=layout)

        return out_pos, out_mass

    def upsample(self, source, resampler=None, keep_mean=False):
        """ Resample an image with the upsample method.

            Upsampling reads out the value of image at the pixel positions of the pm.

            Parameters
            ----------
            source : RealField
                the source image
            keep_mean : bool
                if True, conserves the mean rather than the total mass in the overlapped region.

            Returns
            -------
            A new RealField.

            Notes
            -----
            Note that kernels do not conserve total mass or mean exactly
            by construction due to the sparse sampling, this is particularly bad
            for lanzcos, db, and sym.

            some tests are shown in https://github.com/rainwoodman/pmesh/pull/22
        """
        assert isinstance(source, RealField)

        q = self.mesh_coordinates(dtype='i4')

        # transform from my mesh to source's mesh
        transform = Affine(self.ndim,
                    translate=-source.start,
                    scale=1.0 * source.Nmesh / self.Nmesh,
                    period=source.Nmesh)

        layout = source.pm.decompose(q, smoothing=resampler, transform=transform)
        layout = source.pm.decompose(q, smoothing=1.6, transform=transform)

        f = source.readout(q, resampler=resampler, layout=layout, transform=transform)

        #q1 = layout.exchange(q)
        #v1 = source.readout(q1, resampler=resampler, transform=transform)
        #print(source.start, transform.translate)
        #for a, b in zip(q1, v1):
        #    if all(a == [0, 0]):
        #        print(source.start, a, a * transform.scale + transform.translate, b)
        if not keep_mean:
            f *= (source.pm.Nmesh.prod() / source.pm.BoxSize.prod()) / (self.Nmesh.prod() / self.BoxSize.prod())

        # all are on the grid. NGB is faster, and no need to decompose
        return self.paint(q, mass=f, resampler='nnb', transform=self.affine_grid)

    def downsample(self, source, resampler=None, keep_mean=False):
        """ Resample an image with the downsample method.

            Downsampling paints the value of image at the pixel positions source.

            Parameters
            ----------
            source : RealField
                the source image
            keep_mean : bool
                if True, conserves the mean rather than the total mass in the overlapped region.

            Returns
            -------
            A new RealField.

            Notes
            -----
            Note that kernels do not conserve total mass or mean exactly
            by construction due to the sparse sampling, this is particularly bad
            for lanzcos, db, and sym.

            some tests are shown in https://github.com/rainwoodman/pmesh/pull/22
        """
        assert isinstance(source, RealField)

        q = source.pm.mesh_coordinates(dtype='i4')
        f = source.readout(q, resampler='nnb', transform=source.pm.affine_grid)

        # transform from ssource' mesh to my mesh
        transform = self.affine_grid.rescale(1.0 * self.Nmesh / source.Nmesh)

        if keep_mean:
            f /= (source.pm.Nmesh.prod() / source.pm.BoxSize.prod()) / (self.Nmesh.prod() / self.BoxSize.prod())

        layout = self.decompose(q, smoothing=resampler, transform=transform)
        #q1 = layout.exchange(q)
        #v1 = layout.exchange(f)
        #print(q1, v1)
        return self.paint(q, mass=f, layout=layout, resampler=resampler, transform=transform)


