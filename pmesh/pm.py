import numpy
import pfft
import mpsort
from . import domain
from .window import FindResampler, Affine
from mpi4py import MPI

import numbers # for testing Numbers
import warnings

def is_inplace(out):
    return out is Ellipsis

class slab(numpy.ndarray):
    pass

class slabiter(object):
    def __init__(self, field):
        # we iterate over the slowest axis to gain locality.
        if field.ndim == 2:
            axis = 2
            self.optimized_view = field.value[None, ...]
            self.nslabs = 1
            self.optx = [xx[None, ...] for xx in field.x]
            self.opti = [ii[None, ...] for ii in field.i]
        else:
            axissort = numpy.argsort(field.value.strides)[::-1]
            axis = axissort[0]

            self.optimized_view = field.value.transpose(axissort)
            self.nslabs = field.shape[axis]

            self.optx = [xx.transpose(axissort) for xx in field.x]
            self.opti = [ii.transpose(axissort) for ii in field.i]
        self.x = xslabiter(axis, self.nslabs, self.optx)
        self.i = xslabiter(axis, self.nslabs, self.opti)
        self.axis = axis
        self.Nmesh = field.Nmesh
        self.BoxSize = field.BoxSize

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
    def __init__(self, axis, nslabs, optx):
        self.axis = axis
        self.nslabs = nslabs
        self.optx = optx

    def __iter__(self):
        for irow in range(self.nslabs):
            kk = [x[0] if d != self.axis else x[irow] for d, x in enumerate(self.optx)]
            yield xslab(kk)


class Field(object):
    """ Base class for RealField and ComplexField.

        It only supports those two subclasses.
    """
    def __repr__(self):
        return '%s:' % self.__class__.__name__ + repr(self.value)

    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)

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

    def __eq__(self, other):
        return self[...] == other

    def copy(self):
        other = self.__class__(self.pm)
        other.value[...] = self.value
        return other

    def __init__(self, pm, base=None):
        """ Used internally to add shortcuts of attributes from pm """
        if base is None:
            base = pfft.LocalBuffer(pm.partition)

        self.base = base
        self.pm = pm
        self.partition = pm.partition
        self.BoxSize = pm.BoxSize
        self.Nmesh = pm.Nmesh
        self.ndim = len(pm.Nmesh)

        if isinstance(self, RealField):
            self.value = self.base.view_input()
            self.start = self.partition.local_i_start
            self.cshape = numpy.array([e[-1] for e in self.partition.i_edges], dtype='intp')
            self.x = pm.x
            self.i = pm.i_ind
        elif isinstance(self, ComplexField):
            self.value = self.base.view_output()
            self.start = self.partition.local_o_start
            self.cshape = numpy.array([e[-1] for e in self.partition.o_edges], dtype='intp')
            self.x = pm.k
            self.i = pm.o_ind
            self.real = self.value.real
            self.imag = self.value.imag
            self.plain = self.value.view(dtype=(self.real.dtype, 2))
        else:
            raise TypeError("Olny RealField and ComplexField. No more subclassing");

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

        self.csize = pm.comm.allreduce(self.size)

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
        if isinstance(self, ComplexField):
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
    def slabs(self):
        return slabiter(self)

    def sort(self, out=None):
        warnings.warn("Use ravel instead of sort", DeprecationWarning)
        return self.ravel(out)

    def unsort(self, flatiter):
        warnings.warn("Use unravel instead of unsort", DeprecationWarning)
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
                out.value[...] = self.value
            if isinstance(self, ComplexField) and isinstance(out, RealField):
                self.c2r(out)
            if isinstance(self, ComplexField) and isinstance(out, ComplexField):
                out.value[...] = self.value
            return out

        if isinstance(self, RealField):
            self = self.r2c()

        if isinstance(out, RealField):
            complex = ComplexField(out.pm)
        else:
            complex = out

        complex.value[...] = 0.0

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
            mask = numpy.bitwise_and.reduce(
                 [(n - ii) % n == ii
                    for ii, n in zip(i, complex.Nmesh)])
            slab.imag[mask] = 0

            # remove the nyquist of the output
            # FIXME: the nyquist is messy due to hermitian constraints
            # let's do not touch them till we know they are important.
            mask = numpy.bitwise_or.reduce(
                 [ ii == n // 2
                   for ii, n in zip(i, complex.Nmesh)])
            slab[mask] = 0

            # also remove the nyquist of the input
            mask = numpy.bitwise_or.reduce(
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

        if isinstance(self, ComplexField):
            self = self.c2r()

        if Nmesh is not None:
            # skip resampling if Nmesh is identical to current
            if all(Nmesh == self.pm.Nmesh): Nmesh = None

        if Nmesh is not None:
            pm = self.pm.resize(Nmesh)
            if method is None:
                if any(Nmesh < self.pm.Nmesh): method = 'downsample'
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

class RealField(Field):
    def __init__(self, pm, base=None):
        Field.__init__(self, pm, base)

    def r2c(self, out=None):
        """
        Perform real to complex transformation.

        """
        if out is None:
            out = ComplexField(self.pm)

        if is_inplace(out):
            out = self

        if out is self:
            out = ComplexField(self.pm, base=self.base)

        assert isinstance(out, ComplexField)

        if self.base is out.base:
            self.pm.ipforward.execute(self.base, out.base)
        else:
            self.pm.forward.execute(self.base, out.base)

        # PFFT normalization, same as FastPM
        out.value[...] *= numpy.prod(self.pm.Nmesh ** -1.0)

        return out

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
            out_self.paint(pos, mass=v, resampler=resampler, transform=transform, gradient=gradient, hold=False, layout=layout)

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
        warnings.warn("Use ParticleMesh.paint instead", DeprecationWarning)
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

                r.normp(p=2, zeromode=1) would return |r|^2 but set the zero mode (r == 0) to 1.

            kind : string
                The kind of value in r.
                'relative' means distance from [-0.5 Boxsize, 0.5 BoxSize).
                'index' means [0, Nmesh )
        """
        if out is None:
            out = self.pm.create(mode='real')
        if is_inplace(out):
            out = self

        for x, i, islab, oslab in zip(self.slabs.x, self.slabs.i, self.slabs, out.slabs):
            if kind == 'relative':
                oslab[...] = func(x, islab)
            elif kind == 'index':
                oslab[...] = func(i, islab)
            else:
                raise ValueError("kind is relative, or index")
        return out

    def cdot(self, other):
        return self.pm.comm.allreduce(numpy.sum(self[...] * other[...]))

    def cnorm(self):
        return self.cdot(self)

class ComplexField(Field):
    def __init__(self, pm, base=None):
        Field.__init__(self, pm, base)

    def _expand_hermitian(self, i, y):
        # is the field compressed?
        if self.Nmesh[-1] == self.cshape[-1]:
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
        r = self.pm.create(mode='complex', value=0)

        r.value[...] = (self.value * numpy.conj(other.value))

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
            out = RealField(self.pm, self.base)

        assert isinstance(out, RealField)
        if out.base is not self.base:
            self.pm.backward.execute(self.base, out.base)
        else:
            self.pm.ipbackward.execute(self.base, out.base)

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
            out = ComplexField(v.pm)
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
            kind : string
                The kind of value in k.
                'wavenumber' means wavenumber from [- 2 pi / L * N / 2, 2 pi / L * N / 2).
                'circular' means circular frequency from [- pi, pi).
                'index' means [0, Nmesh )
        """
        if out is None:
            out = self.pm.create(mode='complex')
        if is_inplace(out):
            out = self

        for k, i, islab, oslab in zip(self.slabs.x, self.slabs.i, self.slabs, out.slabs):
            if kind == 'wavenumber':
                oslab[...] = func(k, islab)
            elif kind == 'circular':
                w = [ ki * L / N for ki, L, N in zip(k, self.BoxSize, self.Nmesh)]
                oslab[...] = func(w, islab)
            elif kind == 'index':
                oslab[...] = func(i, islab)
            else:
                raise ValueError("kind is wavenumber, circular, or index")
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

from weakref import WeakValueDictionary

_pm_cache = WeakValueDictionary()

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

    def __init__(self, Nmesh, BoxSize=1.0, comm=None, np=None, dtype='f8', plan_method='estimate', resampler='cic'):
        """ create a PM object.  """
        if comm is None:
            comm = MPI.COMM_WORLD

        self.comm = comm

        if np is None:
            if len(Nmesh) >= 3:
                np = pfft.split_size_2d(self.comm.size)
            elif len(Nmesh) == 2:
                np = [self.comm.size]
            elif len(Nmesh) == 1:
                np = []

        dtype = numpy.dtype(dtype)
        self.dtype = dtype

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

        _cache_args = (tuple(self.Nmesh), tuple(self.BoxSize),
                       MPI._addressof(comm), comm.rank, comm.size,
                       tuple(np), self.dtype, plan_method)

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
            self.procmesh = template.procmesh
        else:
            self.procmesh = pfft.ProcMesh(np, comm=comm)

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

        if template is not None:
            self.forward = template.forward
            self.backward = template.backward
            self.ipforward = template.ipforward
            self.ipbackward = template.ipbackward
        else:
            self.forward = pfft.Plan(self.partition, pfft.Direction.PFFT_FORWARD,
                    bufferin, bufferout, forward,
                    plan_method | pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_TUNE | pfft.Flags.PFFT_PADDED_R2C)
            self.backward = pfft.Plan(self.partition, pfft.Direction.PFFT_BACKWARD,
                    bufferout, bufferin, backward,
                    plan_method | pfft.Flags.PFFT_TRANSPOSED_IN | pfft.Flags.PFFT_TUNE | pfft.Flags.PFFT_PADDED_C2R)

            self.ipforward = pfft.Plan(self.partition, pfft.Direction.PFFT_FORWARD,
                    bufferin, bufferin, forward,
                    plan_method | pfft.Flags.PFFT_TRANSPOSED_OUT | pfft.Flags.PFFT_TUNE | pfft.Flags.PFFT_PADDED_R2C)
            self.ipbackward = pfft.Plan(self.partition, pfft.Direction.PFFT_BACKWARD,
                    bufferout, bufferout, backward,
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
        self.affine = Affine(self.partition.ndim,
                    translate=-self.partition.local_i_start,
                    scale=1.0 * self.Nmesh / self.BoxSize,
                    period = self.Nmesh)

        # Transform from global grid unit to local grid unit.
        self.affine_grid = Affine(self.partition.ndim,
                    translate=-self.partition.local_i_start,
                    scale=1.0,
                    period = self.Nmesh)

        self.resampler = FindResampler(resampler)

        _pm_cache[_cache_args] = self

    def resize(self, Nmesh):
        """
            Create a resized ParticleMesh object, changing the resolution Nmesh.

            Parameters
            ----------
            Nmesh : int or array_like or None
                The new resolution

            Returns
            -------
            A ParticleMesh of the given resolution. If Nmesh is None
            or the same as ``self.Nmesh``, a reference of ``self`` is returned.
        """
        if Nmesh is None: Nmesh = self.Nmesh
        Nmesh_ = self.Nmesh.copy()
        Nmesh_[...] = Nmesh
        if all(self.Nmesh == Nmesh_): return self

        return ParticleMesh(BoxSize=self.BoxSize,
                            Nmesh=Nmesh_,
                            dtype=self.dtype, comm=self.comm)

    def create(self, mode, base=None, value=None, zeros=False):
        """
            Create a field object.

            Parameters
            ----------
            mode : string
                'real' or 'complex'.

            base : object, None
                Reusing the base attribute (physical memory) of an existing field
                object. Provide the attribute, not the field object. (`obj.base` not `obj`)

            value : array_like, None
                initialize the field with the values.
        """

        if zeros:
            warnings.warn("argument zeros is deprecated. use value=0 instead", DeprecationWarning)
            value = 0

        if mode == 'real':
            r = RealField(self, base=base)
        elif mode == 'complex':
            r = ComplexField(self, base=base)
        else:
            raise ValueError('mode must be real or complex')

        if value is not None:
            r[...] = value
        return r

    def generate_whitenoise(self, seed, unitary=False, mean=0, mode='complex', base=None):
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
        complex = ComplexField(self, base=base)
        generate(complex.value, complex.start, complex.Nmesh, seed, bool(unitary))

        # add mean
        def filter(k, v):
            mask = numpy.bitwise_and.reduce([ki == 0 for ki in k])
            v[mask] = mean
            return v

        complex.apply(filter, out=Ellipsis)

        if mode == 'complex':
            return complex
        else:
            return complex.c2r(out=Ellipsis)

    def mesh_coordinates(self, dtype=None):
        coord = numpy.indices(self.partition.local_i_shape, dtype).reshape(self.ndim, -1).T
        source = coord + self.partition.local_i_start
        return source

    def generate_uniform_particle_grid(self, shift=0.5, dtype=None, return_id=False):
        """
            create uniform grid of particles, one per grid point, in BoxSize coordinate.

            Parameters
            ----------
            shift : float, array_like
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
        real = RealField(self)

        _shift = numpy.zeros(self.ndim, dtype)
        _shift[:] = shift
        # one particle per base mesh point
        source = self.mesh_coordinates(dtype)

        source[...] += _shift
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
            out = self.create(mode='real')

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
            out = self.create(mode='real')

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
                    translate=-source.pm.partition.local_i_start,
                    scale=1.0 * source.Nmesh / self.Nmesh,
                    period=source.Nmesh)

        layout = source.pm.decompose(q, smoothing=resampler, transform=transform)
        layout = source.pm.decompose(q, smoothing=1.6, transform=transform)

        f = source.readout(q, resampler=resampler, layout=layout, transform=transform)

        #q1 = layout.exchange(q)
        #v1 = source.readout(q1, resampler=resampler, transform=transform)
        #print(source.pm.partition.local_i_start, transform.translate)
        #for a, b in zip(q1, v1):
        #    if all(a == [0, 0]):
        #        print(source.pm.partition.local_i_start, a, a * transform.scale + transform.translate, b)
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
        transform = Affine(self.ndim,
                    translate=-self.partition.local_i_start,
                    scale=1.0 * self.Nmesh / source.Nmesh,
                    period=self.Nmesh)

        if keep_mean:
            f /= (source.pm.Nmesh.prod() / source.pm.BoxSize.prod()) / (self.Nmesh.prod() / self.BoxSize.prod())

        layout = self.decompose(q, smoothing=resampler, transform=transform)
        #q1 = layout.exchange(q)
        #v1 = layout.exchange(f)
        #print(q1, v1)
        return self.paint(q, mass=f, layout=layout, resampler=resampler, transform=transform)
