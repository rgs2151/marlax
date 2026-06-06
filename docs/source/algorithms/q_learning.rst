Independent Q-Learning
======================

The first method is independent tabular Q-learning with shared cooperative reward.

Loop
----

- Encode the global environment state.
- Choose one action per agent with epsilon-greedy exploration.
- Step all environments as a batch.
- Update each agent's table from the shared reward.
- Repeat with ``jax.lax.scan``.

Use
---

This method is the baseline for tiny discrete worlds. It is not the final coordination
answer, but it gives a fast sanity check for every new environment.
