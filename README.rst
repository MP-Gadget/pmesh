pmesh: Particle Mesh in Python
=============================

The `pmesh` package lays out the fundation of a parallel
Fourier transform particle mesh solver in Python. 

Build Status
------------
.. image:: https://api.travis-ci.org/rainwoodman/pmesh.svg
    :alt: Build Status
    :target: https://travis-ci.org/rainwoodman/pmesh/

This readme file is minimal. We shall expand it.

Reference Manual
----------------

Refer to http://rainwoodman.github.io/pmesh for a full API reference and installation guide.

We recommended working with Anaconda's Python distribution. pmesh is available via the BCCP conda
channel for Anaconda. Installing from the source requires installing pfft from source, and it may
take a while to compile pfft.

Description
-----------

pmesh includes a few software components for building particle mesh simulations
with Python. It consists

- pmesh.domain : a cubinoid domain decomposition scheme in n dimensions. 

- pmesh.pm : a Particle Mesh solver engine, with real-to-complex, complex-to-real
  transforms, transfer functions in real and complex fields, and particle-mesh conversions
  (paint and readout) operations. In order to interface with a higher level differentiable
  modelling package (e.g. abopt [3]_), the back-propagation gradient operators are also implemented.

- pmesh.window : a variety of resampling windows for converting data representation
  between particle and mesh:
  polynomial windows up to cubic. Cloud-In-Cell is the same as the linear window;
  lanczos windows of order 2 and 3; a few wavelet motivated windows (ref needed) that
  perserves the power spectrum to high frequency.

- pmesh.whitenoise : a resolution-invariant whitenoise generator for 2d and 3d fields.

The FFT backend is PFFT [5]_, provided by the pfft-python binding [4]_.
We use MPI to provide parallism (inherited from PFFT). 

Downstream products that uses pmesh includes nbodykit [1]_ and fastpm-python [2]_.

If there are issues starting up a large size MPI job, consult
   http://github.com/rainwoodman/python-mpi-bcast


.. [1] https://github.com/bccp/nbodykit
.. [2] https://github.com/rainwoodman/fastpm-python
.. [3] https://github.com/bccp/abopt
.. [4] https://github.com/rainwoodman/pfft-python
.. [5] https://github.com/mpip/pfft

