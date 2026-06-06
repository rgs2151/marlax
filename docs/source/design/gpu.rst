GPU Runtime
===========

The target runtime is NVIDIA GPU through JAX CUDA wheels.

Local machine
-------------

- Driver: ``595.71.05``.
- CUDA reported by ``nvidia-smi``: ``13.2``.
- GPUs: two NVIDIA GeForce RTX 5090 cards.
- ``nvcc`` is not required for the pip CUDA wheel path.

Install choice
--------------

Use the JAX CUDA 13 pip extra:

.. code-block:: bash

   uv pip install --python /home/dev/miniconda3/envs/marlax/bin/python -e ".[gpu,dev,docs,storage]"

Runtime check
-------------

.. code-block:: bash

   conda run -n marlax python - <<'PY'
   import jax
   print(jax.devices())
   PY

Expected result
---------------

The device list should include CUDA devices. If it only shows CPU, the package install
did not pick up the CUDA-enabled JAX stack.

Docs
----

Sphinx builds force ``JAX_PLATFORMS=cpu`` so autodoc imports do not reserve GPU memory.
