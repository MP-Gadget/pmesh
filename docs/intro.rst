
Introduction
============

PyPM provides a :py:class:`pypm.particlemesh.ParticleMesh` object for solving forces with the particle mesh
method.

ParticleMesh object is a state machine. 

The typical routine for calculating force is

1. Paint particles via :py:meth:`paint` . This step uses the Cloud-in-cell approximation implemented in
   :py:mod:`pypm.cic`
2. Real to Complex transform via :py:meth:`r2c`
3. Complex to Real transform via :py:meth:`c2r`, applying transfer functions. (Refer to :py:class:`pypm.transfer.TransferFunction`)
4. Read out force values     via :py:meth:`readout`. This step uses the trilinear interpolation implemented in
   :py:mod:`pypm.cic`
5. go back to 3, for other force components (eg, x, y, z)

We provide a set of commonly used transfer functions in :py:class:`pypm.transfer.TransferFunction`.

Suppose pm is a :py:class:`pypm.particlemesh.ParticleMesh` object 
and position of particles is stored in :code:`P['Position']` .
Here is an example for calculating gravity:

.. code-block:: python

        smoothing = 1.0 * pm.Nmesh / pm.BoxSize
        # lets get the correct mass distribution with particles on the edge mirrored
        layout = pm.decompose(P['Position'])
        tpos = layout.exchange(P['Position'])
        pm.r2c(tpos, P['Mass'])

        # calculate potential in k-space
        pm.transfer( [
                TransferFunction.RemoveDC,
                TransferFunction.Trilinear,
                TransferFunction.Gaussian(1.25 * smoothing), 
                TransferFunction.Poisson, 
                TransferFunction.Constant(4 * numpy.pi * QPM.G),
                TransferFunction.Constant(pm.Nmesh ** -2 * pm.BoxSize ** 2),
                ])

        for d in range(3):
            pm.c2r( [
                TransferFunction.SuperLanzcos(d), 
                # watch out negative for gravity *pulls*!
                TransferFunction.Constant(- pm.Nmesh ** 1 * pm.BoxSize ** -1),
                TransferFunction.Trilinear,
                ])
            tmp = pm.readout(tpos)
            tmp = layout.gather(tmp, mode='sum')
            P['Accel'][:, d] = tmp
    
