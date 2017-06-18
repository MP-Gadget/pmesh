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

class xslabiter(slabiter):
    def __init__(self, axis, nslabs, optx):
        self.axis = axis
        self.nslabs = nslabs
        self.optx = optx

    def __iter__(self):
        for irow in range(self.nslabs):
            kk = [x[0] if d != self.axis else x[irow] for d, x in enumerate(self.optx)]
            yield kk


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
        r[...] = abs(self[...])
        return r

    def __eq__(self, other):
        return self[...] == other[...]

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
            self.cshape = pm.Nmesh
            self.x = pm.x
            self.i = pm.i_ind
        elif isinstance(self, ComplexField):
            self.value = self.base.view_output()
            self.start = self.partition.local_o_start
            self.cshape = pm.Nmesh.copy()
            self.cshape[-1] = self.cshape[-1] // 2 + 1
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

    def csetitem(self, index, v):
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

        dualv = v
        if localindex is None:
            v = 0
        if duallocalindex is None:
            dualv = 0

        if len(index) == self.ndim + 1 and index[-1] == 1:
            dualv = -dualv
            if localindex is not None and duallocalindex is not None:
                if localindex == duallocalindex:
                    # self dual and imag
                    v = 0
                    dualv = 0
        elif len(index) == self.ndim:
            dualv = numpy.conjugate(dualv)
            if localindex is not None and duallocalindex is not None:
                if localindex == duallocalindex:
                    # self conjugate
                    dualv = dualv.real
                    v = v.real
        if localindex is not None:
            value[localindex] = v
        if duallocalindex is not None:
            value[duallocalindex] = dualv

        return self.pm.comm.allreduce(v)

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
        assert len(flatiter) == self.size

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

    def preview(self, Nmesh=None, axes=None):
        """ gathers the mesh into as a numpy array, with
            (reduced resolution). The result is broadcast to
            all ranks, so this uses Nmesh ** 3 per rank.

            Parameters
            ----------
            Nmesh : int, array_like, None
                The desired Nmesh of the result. Be aware this function
                allocates memory to hold A full Nmesh on each rank.
                None will not resample Nmesh.
            axes : list or None
                list of axes to preserve.

            Returns
            -------
            out : array_like
                An numpy array for the real density field.

        """
        if axes is None: axes = range(self.ndim)
        if not hasattr(axes, '__iter__'): axes = (axes,)
        else: axes = list(axes)

        pm = self.pm.resize(Nmesh)

        out = pm.create(mode='real')
        self.resample(out)

        result = numpy.zeros([out.cshape[i] for i in axes], dtype=pm.dtype)
        local_slice = tuple([out.slices[i] for i in axes])

        if len(axes) != self.ndim:
            removeaxes = set(range(self.ndim)) - set(axes)
            all_axes = list(axes) + list(removeaxes)
            removeaxes = tuple(range(len(all_axes) - len(removeaxes), len(all_axes)))
            result[local_slice] += out[...].transpose(all_axes).sum(axis=removeaxes)
        else:
            result[local_slice] += out[...]

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

    def csum(self):
        """ Collective mean. Sum of the entire mesh. (Must be called collectively)"""
        return self.pm.comm.allreduce(self.value.sum(dtype=self.dtype))

    def cmean(self):
        """ Collective mean. Mean of the entire mesh. (Must be called collectively)"""
        return self.csum() / self.csize

    def paint(self, pos, mass=1.0, resampler=None, transform=None, hold=False, gradient=None, layout=None):
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
            transform = self.pm.affine

        if resampler is None:
            resampler = self.pm.resampler

        resampler = FindResampler(resampler)

        if not hold:
            self.value[...] = 0

        if layout is None:
            return resampler.paint(self.value, pos, mass, transform=transform, diffdir=gradient)
        else:
            localpos = layout.exchange(pos)
            if numpy.isscalar(mass):
                mass = numpy.array(mass)
            if mass.ndim != 0:
                localmass = layout.exchange(mass)
            else:
                localmass = mass
            return self.paint(localpos, localmass,
                    resampler=resampler,
                    transform=transform,
                    hold=hold,
                    gradient=gradient,
                    layout=None)

    def readout(self, pos, out=None, resampler=None, transform=None, gradient=None, layout=None):
        """ 
        Read out from real field at positions

        Parameters
        ----------
        pos    : array_like (, ndim)
            position of particles in simulation  unit
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
            return resampler.readout(self.value, pos, out=out, transform=transform, diffdir=gradient)
        else:
            localpos = layout.exchange(pos)
            localresult = self.readout(localpos, resampler=resampler,
                    transform=transform,
                    gradient=gradient,
                    out=None, layout=None)
            return layout.gather(localresult, out=out)

    def readout_gradient(self, pos, btgrad, resampler=None, transform=None, gradient=None,
            out_self=None, out_pos=None, layout=None):
        """ back-propagate the gradient of readout.

            Returns a tuple of (out_self, out_pos), one of both can be False depending
            on the value of out_self and out_pos.

            Parameters
            ----------
            btgrad : array
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
                out_pos[:, d] *= btgrad

        if out_self is not False:
            if out_self is None:
                out_self = RealField(self.pm)
            if is_inplace(out_self):
                out_self = self

            # watch out: do this after using self, because out_self can be self.
            out_self.paint(pos, mass=btgrad, resampler=resampler, transform=transform, gradient=gradient, hold=False, layout=layout)

        return out_self, out_pos

    def paint_gradient(btgrad, pos, mass=1.0, resampler=None, transform=None, gradient=None,
            out_pos=None, out_mass=None, layout=None):
        """ back-propagate the gradient of paint from self. self contains
            the current gradient.

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
                btgrad.readout(pos, out=out_pos[:, d], resampler=resampler, transform=transform, gradient=d, layout=layout)

        if out_mass is not False:
            if out_mass is None:
                out_mass = numpy.zeros_like(mass)
            if is_inplace(out_mass):
                out_mass = mass
            btgrad.readout(pos, out=out_mass, resampler=resampler, transform=transform, gradient=gradient, layout=layout)

        return out_pos, out_mass

    def c2r_gradient(btgrad, out=None):
        """ Back-propagate the gradient of c2r from self to out """
        out=btgrad.r2c(out)
        # PFFT normalization, same as FastPM
        out.value[...] *= numpy.prod(out.pm.Nmesh ** 1.0)
        return out

    def apply(self, func, kind="relative", out=None):
        """ apply a function to the field.

            Parameters
            ----------
            func : callable
                func(r, v) where r is a list of r values that broadcasts into a full array.
                value of r depends on kind. v is the value of the field on the corresponding locations.
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
        return self.pm.comm.allreduce(numpy.sum(self[...] * other[...], dtype='f8'))

    def cnorm(self):
        return self.cdot(self)

class ComplexField(Field):
    def __init__(self, pm, base=None):
        Field.__init__(self, pm, base)

    def cnorm(self, metric=None):
        r"""compute the norm collectively. The conjugates are added too.

            NORM = Self-conj + lower + upper

            .. math ::

                \sum_{m \in M} (self[m] * conjugate(other[m]) 
                            +   conjugate(self[m]) * other[m])
                             *  0.5  metric(k[m])
        """
        def filter(i, v):
            v = v.copy()
            # if a conjugate is not stored and not self, increase the weight
            # because we shall add it.
            mask = (i[-1] != 0) & (i[-1] != self.Nmesh[-1] // 2)
            v += mask * v
            return v

        def filter2(k, v):
            v = v.copy()
            v.real **= 2
            v.imag **= 2
            if metric is not None:
                k = sum([ki ** 2 for ki in k]) ** 0.5
                v *= metric(k)
            return v

        return self.pm.comm.allreduce(self\
                .apply(filter2)\
                .apply(filter, kind='index', out=Ellipsis)\
                .plain.sum())

    def cnorm_gradient(self, gcnorm, metric=None, out=None):
        def filter2(k, v):
            r = (2 * gcnorm) * v
            if metric is not None:
                k = sum([ki ** 2 for ki in k]) ** 0.5
                r *= metric(k)
            return r

        return self.apply(filter2, out=out).decompress_gradient(out=Ellipsis)

    def cdot(self, other, metric=None):
        r""" Collective inner product between the independent modes of two Complex Fields.

            This is the only reasonable way to define a real inner product of two complex fields.

            when other == self,

            cdot = Self-conj + lower
            cdot = Self-conj + upper

            Parameters
            ----------
            other : ComplexField
                the other field for the inner product
            metric: callable
                metric(k) gives the metric of each mode.

        """
        r = self.copy()
        r.plain[...] *= other.plain

        def filter(i, v):
            v = v.copy()
            mask = numpy.ones_like(v, '?')
            # if a conjugate is stored and not self, reduce the weight
            # because we should have used only one of them (independent)
            mask &= (i[-1] == 0) | (i[-1] == self.Nmesh[-1] // 2)
            mask1 = numpy.ones_like(v, '?')
            for ii, n in zip(i, self.Nmesh):
                mask1 &= (n - ii) % n == ii
            mask &= ~mask1
            v[mask] *= 0.5
            return v

        r.apply(filter, kind='index', out=Ellipsis)

        if metric is not None:
            r.apply(lambda k, v: v * metric(sum(ki**2 for ki in k) ** 0.5), out=Ellipsis)

        return self.pm.comm.allreduce(r.plain.sum(dtype=self.plain.dtype))

    def cdot_gradient(self, gcdot, metric=None):
        """ backtrace gradient of cdot against other. This is a partial gradient.
        """
        r = self * gcdot

        def filter(i, v):
            # changing any number results the same change on an independent mode.
            return v

        r.apply(filter, kind='index', out=Ellipsis)

        if metric is not None:
            r.apply(lambda k, v: v * metric(sum(ki**2 for ki in k) ** 0.5), out=Ellipsis)

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

    def r2c_gradient(btgrad, out=None):
        """ Back-propagate the gradient of r2c to self. """
        out = btgrad.c2r(out)
        # PFFT normalization, same as FastPM
        out.value[...] *= numpy.prod(out.pm.Nmesh ** -1.0)
        return out

    def decompress_gradient(btgrad, out=None):
        """ Back-propagate the gradient of decompress from self to out.
            If I change this mode in the .value array, how many modes are
            actually changed in order to maintain the hermitian?
        """
        if out is None:
            out = ComplexField(btgrad.pm)
        if is_inplace(out):
            out = btgrad

        for i, a, b in zip(btgrad.slabs.i, out.slabs, btgrad.slabs):
            # modes that are self conjugates do not gain a factor
            mask = numpy.ones(a.shape, '?')
            for ii, n in zip(i, btgrad.Nmesh):
               mask &= (n - ii) % n == ii
            a[~mask] = 2 * b[~mask]
            a[mask] = b[mask]
        return out

    def apply(self, func, kind="wavenumber", out=None):
        """ apply a function to the field, in-place.

            Parameters
            ----------
            func : callable
                func(k, v) where k is a list of k values that broadcasts into a full array.
                value of k depends on kind. v is the corrsponding value of field.
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
        dtype of the buffers

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
        else:
            raise ValueError("dtype must be f8 or f4")

        self.procmesh = pfft.ProcMesh(np, comm=comm)
        self.Nmesh = numpy.array(Nmesh, dtype='i8')
        self.ndim = len(self.Nmesh)
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

        self.resampler = FindResampler(resampler)

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

    def create(self, mode, base=None, zeros=False):
        """
            Create a field object.

            Parameters
            ----------
            mode : string
                'real' or 'complex'.

            base : object, None
                Reusing the base attribute (physical memory) of an existing field
                object. Provide the attribute, not the field object.
        """

        if mode == 'real':
            r = RealField(self, base=base)
        elif mode == 'complex':
            r = ComplexField(self, base=base)
        else:
            raise ValueError('mode must be real or complex')
        if zeros:
            r[...] = 0
        return r

    def generate_whitenoise(self, seed, unitary=False, mode='complex', base=None):
        """ Generate white noise to the field with the given seed.

            The scheme is supposed to be compatible with Gadget when the field is three-dimensional.

            Parameters
            ----------
            seed : int
                The random seed
            unitary : bool
                True to generate a unitary white noise where the amplitude is fixed to 1 and
                only the phase is random.
        """
        from .whitenoise import generate
        complex = ComplexField(self, base=base)
        generate(complex.value, complex.start, complex.Nmesh, seed, bool(unitary))

        if mode == 'complex':
            return complex
        else:
            return complex.c2r(out=Ellipsis)

    def generate_uniform_particle_grid(self, shift=0.5, dtype=None):
        """
            create uniform grid of particles, one per grid point on the basepm mesh
            this shall go to pmesh.
        """
        if dtype is None: dtype == self.dtype
        real = RealField(self)

        _shift = numpy.zeros(self.ndim, dtype)
        _shift[:] = shift
        # one particle per base mesh point
        source = numpy.zeros((real.size, self.ndim), dtype=dtype)

        for d in range(len(real.shape)):
            real[...] = 0
            for xi, slab in zip(real.slabs.i, real.slabs):
                slab[...] = (xi[d] + 1.0 * _shift[d]) * (real.BoxSize[d] / real.Nmesh[d])
            source[..., d] = real.value.flat
        return source

    def decompose(self, pos, smoothing=None):
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

        # Transform from simulation unit to global grid unit.
        def transform0(x):
            return self.affine.scale * x

        return self.domain.decompose(pos, smoothing=smoothing,
                transform=transform0)
