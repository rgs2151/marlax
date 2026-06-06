Architecture
============

MARLAX should stay functional at the core and object-shaped at the edges.

Core principles
---------------

- Environments are pure JAX transitions.
- Training loops use ``jax.lax.scan``.
- Batches are first-class; one env is just batch size one.
- Algorithms operate on explicit state objects.
- Experiment output goes through one storage layer.
- Analysis code reads storage; it does not reach into training internals.

Package layout
--------------

- ``marlax.envs``: cooperative multi-agent environments.
- ``marlax.agents``: model-free and model-based methods.
- ``marlax.storage``: run stores, schemas, and array writers.
- ``marlax.training``: reusable rollout and update loops.
- ``marlax.analysis``: storage-native metrics, summaries, and plotting helpers.

Method zoo direction
--------------------

- Start with independent tabular Q-learning.
- Add centralized tabular Q-learning for tiny discrete games.
- Add value iteration and planning for model-based baselines.
- Add deep Q-learning once environment and storage contracts are stable.
- Add actor-critic methods after discrete control is solid.

Environment zoo direction
-------------------------

- Start with cooperative gridworlds.
- Add matrix games for algorithm sanity checks.
- Add communication games.
- Add navigation and foraging tasks.
- Add wrappers for observation masking, reward shaping, and curriculum regimes.
