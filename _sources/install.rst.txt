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

    conda install -c bccp pfft-python runtests

in development mode from git cloned version of source code.

.. code-block:: sh

    git clone https://github.com/rainwoodman/pmesh
    cd pmesh

The development shall ideally be test driven. Write test cases
in the tests directories in the source code, then invoke them with

.. code-block:: sh

    python run-tests.py pmesh/tests/test_....py::test_...

or with a single rank

.. code-block:: sh

    python run-tests.py --single pmesh/tests/test_....py::test_...

Replace `...` with names of files and functions.

`run-tests.py` takes care of building and installation before testing.

