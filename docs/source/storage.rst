Saving and restoring
====================

MARLAX stores numeric arrays in compressed NPZ files and configuration in JSON.
``numpy.load`` uses ``allow_pickle=False``. Model loading never executes serialized
Python objects.

The examples below continue from the training run in :doc:`getting_started`.

Inference policy
----------------

.. code-block:: python

   mx.save_policy("runs/my_policy", run.policy, env.spec)
   policy = mx.load_policy("runs/my_policy")

A policy directory contains ``variables.npz`` (named ``q`` array) and a
``manifest.json`` describing the format version, environment specification,
sharing, model initialization, dtype, shape, and greedy tie rule. It contains no
optimizer, exploration schedule, or rollout state. Version 1 supports QTable
policies.

Full training checkpoint
------------------------

.. code-block:: python

   mx.save_checkpoint("runs/checkpoint", run.state)
   template = mx.initialize(env, params, agent, method, seed=0, num_envs=64)
   state = mx.load_checkpoint("runs/checkpoint", template)
   continued = mx.train(
       env, params, agent, method,
       state=state, env_steps=64_000,
   )

A checkpoint has ``state.npz`` containing ordered numerical PyTree leaves and a
small format manifest. The template must be constructed with the same environment,
model, method, world count, and package version. Static agent descriptions come
from the template. Keep the run's resolved configuration alongside the checkpoint.

The runner uses serializable uint32 JAX PRNG keys and stores pending SARSA actions
as well as counters and environments. This supports continuation without resetting
episodes or silently resampling the next action. Numerical reproducibility across
different hardware or library versions is not guaranteed.

Run directory
-------------

.. code-block:: text

   runs/coop_grid_td/RUN_ID/
     config.json
     provenance.json
     source.tar.gz
     metrics.jsonl
     checkpoint/
       state.npz
       manifest.json
     policy/
       variables.npz
       manifest.json
     evaluation.json

The recipe snapshots tracked and untracked source/configuration documents before
training into ``source.tar.gz``, and records the source revision and dirty status
in provenance. This preserves the actual code even when working on uncommitted
changes. Evaluation includes a SHA-256 fingerprint of the policy array. Run
outputs are ignored by Git.

Load the example policy
-----------------------

The repository includes a trained cooperative-grid Q-table. Load its arrays
into a ``FrozenPolicy`` for evaluation:

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np
   import marlax as mx
   from marlax.models import QTable

   with np.load("stores/coop_grid_q_learning/latest/q_values.npz",
                allow_pickle=False) as saved:
       policy = mx.FrozenPolicy(mx.Agent(QTable()), jnp.asarray(saved["q"]))

This file contains inference parameters. To resume training, use a full
checkpoint containing runner state and random keys.
