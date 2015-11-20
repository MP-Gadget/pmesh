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

API Reference
-------------
Refer to http://rainwoodman.github.io/pmesh for a full API reference.


Description
-----------

The domain decomposition is 2-d (pencil/stencils). [pmesh.domain]
FFT is supported by pfft-python. 
Currently the conversion between particle and mesh we only implement
the linear window function (Cloud-in-Cell). Plans are to implement
other windows, e.g. truncated lanczos or some wavelet motivated windows
that acts as an apodization filter to suppress aliasing effect.

Currently, a particle mesh gravity solver is in utils/gravpm.py . 
We have compared that the force output at first time step agrees with 
the long range force calculation in MP-Gadget3.

We also provide a simple (long range) gravitational strain calculator in utils/strain.py .
The calculator have been used to calculate the strain tensor for the RunPB dark matter simulations 
(2048^3 particles in a 1380 Mpc/h box), on 576 MPI ranks at Edison.

There is a power-spectrum calculator in utils/powerspectrum.py

If there are issues starting up a large size MPI job, consult 
   http://github.com/rainwoodman/python-mpi-bcast


Yu Feng
