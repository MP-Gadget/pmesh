.. _introduction:

Introduction
============

pmesh provides a :py:class:`pmesh.pm.ParticleMesh` object for solving forces with the particle mesh / spectrum
method.

We represent the fields as :py:class:`pmesh.pm.ComplexField` and :py:class:`pmesh.pm.RealField`.

The typical routine for calculating force is

1. Distrubte particles to the correct domain via :py:meth:`pmesh.pm.ParticleMesh.decompose`. In this step,
   ghost particles are created automatically. The result is a :py:meth:`pmesh.domain.Layout` object.

2. Create a :py:class:`pmesh.pm.RealField` object and paint particles via :py:meth:`pmesh.pm.RealField.paint`;
   supply the domain decomposition layout as an argument to take care of the ghosts.

3. Transform to spectrum space via a real-to-complex :py:meth:`pmesh.pm.RealField.r2c`, the result is a :py:class:`pmesh.pm.ComplexField`. The operation
   can be made in-place by setting `out` argument to `Ellipsis`.

5. Apply transfer functions to obtain the force, via :py:meth:`pmesh.pm.ComplexField.apply`. Provide transfer function as `function((kx, ky, kz), original_value)`.

6. Transform to configuration space via a complex-to-real :py:meth:`pmesh.pm.ComplexField.c2r`.

6. Readout force values via :py:meth:`pmesh.pm.RealField.readout`.
   supply the domain decomposition layout as an argument to take care of the ghosts.

This is a fairly convoluted process; but it truthfully represents the level of complexity of distributed computation introduces.
We may provide a higher level interface in the future.

We provide an example to illustrate the process. Suppose pm is a :py:class:`pmesh.pm.ParticleMesh` object 
and position of particles is stored in :code:`P['Position']` .

Here is an example for calculating linear order displacement from perturbative growth of large scale structure:

.. code-block:: python

        pm = ParticleMesh(BoxSize, Nmesh=[Nmesh, Nmesh, Nmesh])

        smoothing = 1.0 * pm.Nmesh / pm.BoxSize

        # lets get the correct mass distribution with particles on the edge mirrored
        layout = pm.decompose(P['Position'])

        density = pm.create(mode='real')
        density.paint(P['Position'], weight=P['Mass'], layout=layout)

        def potential_transfer_function(k, v):
            k2 = k.normp(zeromode=1)
            return v / (k2)

        pot_k = density.r2c(out=Ellipsis)\
                       .apply(potential_transfer_function, out=Ellipsis)

        for d in range(3):
            def force_transfer_function(k, v, d=d):
                return ki[d] * 1j * v

            force_d = pot_k.apply(force_transfer_function) \
                 .c2r(out=Ellipsis)

            P['Accel'][:, d] = force_d.readout(P['Position'], layout=layout)
