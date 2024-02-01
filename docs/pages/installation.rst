Installation
============

Installing with pip
-----------------------
`cortecs` is distributed on `PyPI <https://pypi.org/>`_. It can be installed with

.. code-block:: bash

    pip install cortecs

Installing from source
-----------------------

`cortecs` is developed on `GitHub <https://github.com/arjunsavel/cortecs>`_.
If you received the code as a tarball or zip, feel free to skip the first two lines; they essentially download the source code.
We recommend running the below lines in a fresh `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_ environment
to avoid package dependency isues.

.. code-block:: bash

    python3 -m pip install -U pip
    python3 -m pip install -U setuptools setuptools_scm pep517
    git clone https://github.com/arjunsavel/cortecs.git
    cd cortecs
    python3 -m pip install -e .

If you plan on using the neural network-based compression method of `cortecs`, run the last line as

.. code-block:: bash

    python3 -m pip install -e .[neural_networks]



We plan to distribute `cortecs` through `conda` as well at a later date.

Test the installation
---------------------

To ensure that the installation has been performed smoothly, run

.. code-block:: bash

    python3 -c 'import cortecs'

To ensure the code is working as expected, feel free to run the unit and integration tests included with the package.
From the outermost :code:`cortecs` directory, run

.. code-block:: bash

    python3 -m unittest discover src/cortecs/tests

These tests should only take a couple of minutes to run on a laptop.
