Getting started
===============

Installation
------------

Use Python 3.11 or newer. From the repository root:

.. code-block:: bash

   pip install -e .

For a CUDA 13 installation, add the ``gpu`` extra. GPU installation is optional;
training and inference also run on CPU.

.. code-block:: bash

   pip install -e ".[gpu]"

To use conda, create an environment from ``environment.yml``, then install
the package with pip. See :doc:`inventory` for optional installation extras.

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

This example uses 64,000 world transitions. For a longer training run, use the
``demo`` profile in :doc:`experiments`, which also saves configuration,
checkpoints, policies, and evaluation results.
