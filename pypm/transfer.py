import numpy
from mpi4py import MPI
try:
    from numba import jit
except ImportError:
    jit = None
    pass
class TransferFunction:
    """ these a a function of Window Transfer functions used by PM.
        they take the fourier-space field complex and the dimensionless circular frequency 
        as inputs; complex is modified in-place.

        some functions are factories: they return an actually window with the
        given parameter.

        Working out the dimension of the input is important.
            
        the output of Poisson introduces a dimension to rho_k!
        the output of SuperLanzcos introduces a dimension to rho_k!

        w is a tuple of (w0, w1, w2, ...)
        w is in circular frequency units. The dimensionful k is w * Nmesh / BoxSize 
        (nyquist is at about w = pi)
        they broadcast to the correct shape of complex. This is to reduce
        memory usage somewhat.
        complex is modified in place.

    """
    @staticmethod
    def NormalizeDC(comm, complex, w):
        """ removes the DC amplitude. This effectively
            divides by the mean
        """
        ind = []
        value = 0.0
        found = True
        for wi in w:
            if (wi != 0).all():
                found = False
                break
            ind.append((wi == 0).nonzero()[0][0])
        if found:
            ind = tuple(ind)
            value = numpy.abs(complex[ind])
        value = comm.allreduce(value, MPI.SUM)
        complex[:] /= value
    @staticmethod
    def RemoveDC(complex, w):
        ind = []
        for wi in w:
            if (wi != 0).all():
                return
            ind.append((wi == 0).nonzero()[0][0])
        ind = tuple(ind)
        complex[ind] = 0
    @staticmethod
    def Trilinear(complex, w):
        for wi in w:
            # convert to 
            tmp = numpy.sinc(wi / (2 * numpy.pi)) ** 2
            complex[:] /= tmp
    @staticmethod 
    def SuperLanzcos(dir, order=3):
        """ Super Lanzcos kernel of order 3.
            is complex * i * w in a fancier way.

            Notice that for differentiation, one actually wants
            complex * i * k which is
            complex * i * w * Nmesh / BoxSize
        """
        def SuperLanzcosDir(complex, w):
            wi = w[dir] * 1.0
        #    tmp = (1. / 594 * 
        #       (126 * numpy.sin(wi) + 193 * numpy.sin(2 * wi) + 142 * numpy.sin (3 *
        #           wi) - 86 * numpy.sin(4 * wi)))
            tmp = 1 / 6.0 * (8 * numpy.sin (wi) - numpy.sin (2 * wi))
            if order == 0:
                complex *= wi * 1j
            else:
                complex[:] *= tmp * 1j
        return SuperLanzcosDir
    @staticmethod
    def Gaussian(smoothing):
        """ smoothing is in mesh units;
            Notice that this is different from the usual PM split convention.
            (used in Gadget2/3)
            The PM split is cut = sqrt(0.5) * smoothing
        """
        sm2 = smoothing ** 2

        def GaussianS(complex, w):
            w2 = 0
            for wi in w:
                wi2 = wi ** 2
                complex *= numpy.exp(-0.5 * wi2 * sm2)
        return GaussianS
    @staticmethod
    def Constant(C):
        def Constant(complex, w):
            complex *= C
        return Constant
    @staticmethod
    def Inspect(name, *indices):
        """ inspect the complex array at given indices
            mostly for debugging.
        """
        def Inspect(comm, complex, w):
            V = ['%s = %s' %(str(i), str(complex[tuple(i)])) for i in indices]
            print name, ','.join(V)
        return Inspect

    @staticmethod
    def PowerSpectrum(wout, psout):
        """ calculate the power spectrum.
            This shall be done after NormalizeDC and RemoveDC
        """
        def PS(comm, complex, w):
            wedges = numpy.linspace(0, numpy.pi, wout.size + 1, endpoint=True)
            w2edges = wedges ** 2
            w2 = 0.0
            for wi in w:
                w2 = w2 + wi ** 2
            dig = numpy.digitize(w2.flat, w2edges)
            w2 = numpy.bincount(dig, weights=w2.flat, minlength=wout.size + 2)[1: -1]
            w2 = comm.allreduce(w2, MPI.SUM)

            N = numpy.bincount(dig, minlength=wout.size + 2)[1: -1]
            N = comm.allreduce(N, MPI.SUM)

            P = numpy.abs(complex) ** 2
            P = numpy.bincount(dig, weights=P.flat, minlength=wout.size + 2)[1: -1]
            P = comm.allreduce(P, MPI.SUM)

            psout[:] = P / N 
            wout[:] = (w2 / N) ** 0.5
        return PS

    @staticmethod
    def Laplace(complex, w):
        """ 
            Take the Laplacian k-space: complex *= -w2

            where this function performs only the -w **-2 part.

            Note that k = w * Nmesh / BoxSize, thus the usual laplacian is
           
            - k ** 2 * complex = (Nmesh / BoxSize) ** 2 (-w**2) * complex

        """
        w2 = 0.0
        for wi in w:
            w2 = w2 + wi ** 2
        w2 *= -1
        complex[:] *= w2

    @staticmethod
    def Poisson(complex, w):
        """ 
            Solve Poisson equation in k-space: complex /= -w2

            Notes about gravity:

            gravity is 
            
            pot_k = -4pi G delta_k * k **-2

            where k = w * Nmesh / BoxSize
            hence
            pot_k = -4pi G delta_k * w **-2  * (Nmesh / BoxSize) ** -2 
                  = -4pi G delta_k * w **-2 * Nmesh ** -2 * BoxSize ** -2

            where this function performs only the -w **-2 part.
        """
        w2 = 0.0
        for wi in w:
            w2 = w2 + wi ** 2
        w2[w2 == 0] = numpy.inf
        w2 *= -1
        complex[:] /= w2

