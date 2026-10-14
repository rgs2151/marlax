Getting started
===============

Installation
------------

Use Python 3.11 or newer. From the repository root:

.. code-block:: bash

   pip install -e ".[dev,docs,viz]"

For a CUDA 13 installation, add the ``gpu`` extra. GPU installation is optional;
all package functions and small tests can run on CPU.

.. code-block:: bash

   pip install -e ".[gpu,dev,docs,viz]"
   CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
       python experiments/coop_grid_td/run.py train --profile smoke

The visible device is renumbered inside the process: physical GPU 1 appears as
JAX device 0. The library does not set device visibility or allocator settings.
Set them before starting Python. ``environment.yml`` remains a convenient conda
Python environment; installing the package is a separate step.

A small run
-----------

.. code-block:: python

   import marlax as mx
   from marlax.envs import CooperativeGrid
   from marlax.methods import ExpectedSARSA
   from marlax.models import QTable

   env = CooperativeGrid(size=5, num_agents=2)
   params = env.default_params()
   agent = mx.Agent(QTable(), sharing="independent")
   method = ExpectedSARSA(epsilon_steps=64_000)
   run = mx.train(
       env, params, agent, method,
       seed=0, num_envs=64, env_steps=64_000,
   )
   report = mx.evaluate(
       run.policy, env, params,
       initial_states=env.all_starts(), max_steps=params.time_limit,
   )

This is a smoke-scale learning budget, not a promise of a good policy. The
experiment recipe provides a longer ``demo`` profile and saves configuration,
training state, inference variables, and evaluation results.

Documentation and checks
------------------------

.. code-block:: bash

   JAX_PLATFORMS=cpu python -m pytest -q
   JAX_PLATFORMS=cpu sphinx-build -W --keep-going -b html docs/source docs/build/html
   python -m http.server 8001 --directory docs/build/html

Documentation uses reStructuredText, Sphinx autodoc, and the standard Alabaster
theme. GitHub Actions builds it with saved notebook outputs, checks local links,
and publishes successful main-branch builds at ``/marlax/``. The existing
gallery is available at ``/marlax/gallery/`` in the same Pages deployment. Pull requests build
and validate the site but do not deploy it.
