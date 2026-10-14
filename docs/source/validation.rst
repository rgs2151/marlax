Evaluation results
==================

Cooperative grid
----------------

The following runs use a 5×5 grid with two agents and independent Q-tables.
Each method trains for 12,288,000 world transitions across 2,048 parallel worlds
with seed 0, alpha 0.03, and gamma 0.97. Epsilon decreases linearly from 1.0 to
0.02 over the training budget.

Evaluation uses greedy actions across all 3,750 ordered start-position/target
combinations. All three policies collect the target in every case, with no
wrong-target outcomes or time-limit truncations.

.. list-table::
   :header-rows: 1

   * - Method
     - Collection rate
     - Mean episode steps
     - Training and evaluation (seconds)
   * - Q-learning
     - 100%
     - 4.984
     - 6.07
   * - SARSA
     - 100%
     - 4.984
     - 6.48
   * - Expected SARSA
     - 100%
     - 4.984
     - 5.98

Timings were recorded on an NVIDIA GeForce RTX 5090 and include compilation
and file output. These single-seed results describe the example configuration;
they do not measure performance across seeds or relative to CPU execution.
Configurations and measurements are available in
:download:`validation.json <../../experiments/coop_grid_td/validation.json>`.

Run the experiment from the repository root:

.. code-block:: bash

   python experiments/coop_grid_td/run.py train --method q_learning --profile demo

Select ``sarsa`` or ``expected_sarsa`` to reproduce the other configurations.
See :doc:`experiments` for evaluation, plotting, and checkpoint commands.

Included pretrained policy
--------------------------

The Q-table in ``stores/coop_grid_q_learning/latest/q_values.npz`` achieves
100% collection over the same 3,750 cases, with a mean episode length of
5.1296 steps. Evaluate it with:

.. code-block:: bash

   python experiments/coop_grid_td/run.py evaluate-saved

Custom environment
------------------

The two-agent game in ``examples/coordination.py`` achieves full coordination
across 256 evaluation episodes after 4,096 training transitions on CPU, using
shared parameters. See :doc:`notebooks/06_custom_environment` for the environment
implementation and training procedure.
