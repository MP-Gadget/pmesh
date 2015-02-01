# pypm: Particle Mesh in Python


This readme file is minimal. We shall expand it.

The `pypm` package provides a general purpose particle mesh solver in Python, under the module
`pypm.particlemesh`.
`pypm` requires `mpi4py`. 


Currently, a particle mesh gravity solver is in utils/gravpm.py . 
We have compared that the force output at first time step agrees with the long range force calculation in MP-Gadget3.

We also provide a simple (long range) gravitational strain calculator in utils/strain.py .
The calculator have been used to calculate the strain tensor for the RunPB dark matter simulations 
(2048**3 particles in a 1380 Mpc/h box), on 576 MPI ranks at Edison.

There is a power-spectrum calculator in utils/powerspectrum.py

If there are issues starting up a large size MPI job, consult 
   http://github.com/rainwoodman/mpiimport 


Yu Feng
