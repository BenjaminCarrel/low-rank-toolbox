Installation
============

For Users
---------

Once the package is published, installation will be straightforward:

**Via pip** (when available on PyPI):

.. code-block:: bash

   pip install lowrank

**Via conda** (when available on conda-forge):

.. code-block:: bash

   conda install -c conda-forge lowrank

.. note::
   The package is currently under development. For now, please use the developer installation method below.

For Developers
--------------

To contribute to the development or modify the package:

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/BenjaminCarrel/low-rank-toolbox.git
      cd low-rank-toolbox

2. **Create the conda environment:**

   .. code-block:: bash

      conda env create -f environment.yml
      conda activate low-rank-dev

   This will install all necessary dependencies, including development tools like pytest.

3. **Install in editable mode:**

   .. code-block:: bash

      pip install -e .

   Any changes you make in the ``src/`` directory will be immediately available without needing to reinstall.

Verifying the Installation
---------------------------

Run the comprehensive test suite (1000+ tests):

.. code-block:: bash

   pytest

All tests should pass. If you encounter any issues, please `open an issue <https://github.com/BenjaminCarrel/low-rank-toolbox/issues>`_.

Requirements
------------

**Core Dependencies:**

* Python >= 3.10
* NumPy >= 1.21
* SciPy >= 1.7

**Development Dependencies:**

* pytest >= 8.3.5
* sphinx >= 7.0 (for documentation)
* sphinx_rtd_theme (for documentation)

See ``environment.yml`` and ``pyproject.toml`` for complete dependency information.
