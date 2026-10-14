Saving and restoring
====================

Use compressed NPZ for numeric arrays and JSON for configuration. The current
small tables do not need Zarr, pickle, a database, or a checkpoint service.
``numpy.load`` uses ``allow_pickle=False``. Model loading never executes serialized
Python objects.

Inference policy
----------------

.. code-block:: python

   mx.save_policy("runs/my_policy", run.policy, env.spec)
   policy = mx.load_policy("runs/my_policy")

A policy directory contains ``variables.npz`` (named ``q`` array) and a
``manifest.json`` describing the format version, environment specification,
sharing, model initialization, dtype, shape, and greedy tie rule. It contains no
optimizer, exploration schedule, or rollout state. Version 1 supports QTable
policies only. It is a Python inference format, not the planned browser bundle.

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
from the template; there is no automatic schema migration or compatibility
loader. Keep the run's resolved config alongside the checkpoint. Different
architectures are new runs, not resumptions.

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
outputs are ignored by Git. Selected evaluation summaries can be documented
separately; training never overwrites the historical checkpoint.

Original saved Q-table
----------------------

The original NPZ can be supplied explicitly to the new policy constructor because
its observation/action ordering is unchanged:

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np
   import marlax as mx
   from marlax.models import QTable

   with np.load("stores/coop_grid_q_learning/latest/q_values.npz",
                allow_pickle=False) as saved:
       policy = mx.FrozenPolicy(mx.Agent(QTable()), jnp.asarray(saved["q"]))

This is inference from known arrays, not restoration of the old training run.
The original file has no runner RNG or pending actions. New training also changes
duplicate-update and truncation semantics. The historical file remains untouched.
