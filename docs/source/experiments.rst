Experiments
===========

Run from the repository root:

.. code-block:: bash

   python experiments/coop_grid_td/run.py train --method q_learning --profile smoke
   python experiments/coop_grid_td/run.py train --method sarsa --profile demo
   python experiments/coop_grid_td/run.py train --method expected_sarsa --profile demo
   python experiments/coop_grid_td/run.py evaluate --run runs/coop_grid_td/RUN_ID
   python experiments/coop_grid_td/run.py plot --run runs/coop_grid_td/RUN_ID
   python experiments/coop_grid_td/run.py resume --run runs/coop_grid_td/RUN_ID --env-steps 2048000
   python experiments/coop_grid_td/run.py evaluate-saved

The smoke profile uses 32 worlds and 1,024 transitions. The demo profile uses
2,048 worlds and 12,288,000 transitions. ``--env-steps`` overrides the additional
world-transition budget. The actual budget rounds up to a full vector step.
``--output`` selects a fresh run directory; existing directories are not reused.

Resuming restores the original method configuration, RNG state, world state,
current observations, pending behavior actions, and counters. Exploration continues
from its original schedule rather than starting again. Resume writes a new run;
it does not overwrite the source checkpoint.

The recipe selects public package components. It contains no copy of environment
physics or TD equations. Training and Python evaluation use the same environment
step. Evaluation freezes behavior and excludes completed episodes from subsequent
rewards and metrics.

Metrics
-------

``env_steps`` counts world transitions, ``updates`` counts vector learning steps,
and ``episodes`` counts completed worlds. Chunk means report ``reward_per_step``,
``mean_abs_td``, and ``epsilon``. Evaluation reports per-episode event fractions,
mean return, mean episode length, termination/truncation fractions, and an
``unfinished_rate`` if its explicit rollout budget ends early.

For the cooperative grid, event fractions include ``activated_rate``,
``collected_rate``, and ``wrong_target_rate``. Activation overlaps later outcomes;
it is not a disjoint failure category. ``collected_rate``, ``wrong_target_rate``,
``truncation_rate``, and ``unfinished_rate`` form the outcome partition here.
Team return averages the duplicated per-agent reward rather than summing it.

The built-in exhaustive scenario set covers all 3,750 start-position/target
combinations for the 5×5 two-agent task. It does not demonstrate generalization to
other sizes or observation schemas. A single-seed run is a development check;
scientific algorithm comparisons require a separate multi-seed experiment.

The recipe never retrains when evaluating, loading, or inspecting a saved policy.
It does not write ``site/``, ``stores/``, or the old diagnostic plots.
The independent ``plot`` command writes a small SVG under the run's ``plots/``
directory. It separates training reward per step from evaluation outcome fractions.
