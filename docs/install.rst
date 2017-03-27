Installation
============

Anaconda
--------

With anaconda, a binary can be obtained from the BCCP channel via

.. code-block:: sh

    conda install -c bccp pmesh


PyPI
----

.. code-block:: sh

    pip install pfft-python pmesh

pmesh depends on `pfft-python <http://github.com/rainwoodman/pfft-python>`_ for fast fourier
transformation.


For Development
---------------

Recommended development environment is anaconda. First install pfft-python

.. code-block:: sh

    conda install -c bccp pfft-python

Then install pmesh in development mode from git cloned version of source code.

.. code-block:: sh

    git clone https://github.com/rainwoodman/pmesh
    cd pmesh
    pip install -e .

I recommended running the test suites

.. code-block:: sh

    python runtests.py  --mpirun

No tests shall fail.

