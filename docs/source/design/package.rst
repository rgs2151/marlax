Package And Environment
=======================

The package is installed from ``pyproject.toml``. Conda owns the Python interpreter and
uv owns Python package installation.

Environment rule
----------------

- The canonical conda env name is ``marlax``.
- The conda env contains only Python, pip, and uv.
- Project dependencies come from ``pyproject.toml``.
- CUDA-enabled JAX is installed through the ``gpu`` extra.
- Local development uses editable install.

Commands
--------

.. code-block:: bash

   conda env create -f environment.yml
   conda run -n marlax uv pip install --python /home/dev/miniconda3/envs/marlax/bin/python -e ".[gpu,dev,docs,storage,viz]"
   conda run -n marlax python -m pytest -q
   conda run -n marlax make -C docs html

Dependency extras
-----------------

- ``dev``: tests and developer checks.
- ``docs``: Sphinx documentation.
- ``gpu``: NVIDIA CUDA 13 JAX wheels.
- ``storage``: Zarr-backed experiment storage.
- ``viz``: matplotlib, seaborn, pillow, and tqdm for gallery assets.
- ``all``: development setup with GPU, docs, tests, storage, and visualization.
