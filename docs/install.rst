Installation
============

PyPM depends on `pfft-python <http://github.com/rainwoodman/pfft-python>`_ for fast fourier
transformation.

Therefore, first install pfft-python via git (easy_install is not recommented yet)

.. code-block:: sh

    git clone http://github.com/rainwoodman/pfft-python
    cd pfft-python
    python setup.py install --user
    cd ..

Next, install pypm itself

.. code-block:: sh

    git clone http://github.com/rainwoodman/pypm
    cd pypm
    python setup.py install
    cd ..



