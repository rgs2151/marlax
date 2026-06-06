Q-Learning First Pass
=====================

The first algorithm is independent tabular Q-learning with shared cooperative reward.

What it means
-------------

- Each agent owns one table of shape ``num_states x num_actions``.
- All agents observe the same encoded global state.
- Each agent chooses its own action with epsilon-greedy exploration.
- The environment returns a shared team reward copied across agents.
- Updates run in batched JAX over parallel environments.

Why this first
--------------

- It is simple enough to test completely.
- It exposes the state/action/env contracts early.
- It gives fast baselines for tiny cooperative tasks.
- It makes later model-free methods easier to compare.

Known limits
------------

- Tabular state spaces grow quickly with grid size and agent count.
- Independent learners are non-stationary from each agent's perspective.
- Cooperative coordination can fail without centralized training or richer state.

Next improvements
-----------------

- Add centralized joint-action Q-learning for small tasks.
- Add observation encoders instead of one hard-coded state id.
- Add replay-free and replay-based training variants.
- Add per-run metrics into Zarr storage.
