Getting Started
===============

MARLAX uses conda for the Python environment and uv for package installation.

.. code-block:: bash

   conda env create -f environment.yml
   conda run -n marlax uv pip install --python /home/dev/miniconda3/envs/marlax/bin/python -e ".[gpu,dev,docs,storage,viz]"

Build the site:

.. code-block:: bash

   conda run -n marlax make -C docs html

Regenerate the first gallery asset:

.. code-block:: bash

   conda run -n marlax python visualize/cooperative_grid.py

The gallery scripts use progress bars while rendering so slow assets are obvious.
