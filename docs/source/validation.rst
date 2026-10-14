Migration validation
====================

The Python migration preserves the historical policy's state encoding and grid
movement/reward rules. Evaluating the original saved table through the new JAX
step gives 100% collection across all 3,750 ordered start-position/target cases,
with mean episode length 5.1296 steps. These match the original stored evaluation.

The small suite covers activation and collection, split penalties, wrong targets,
time limits, averaged duplicate updates, all three TD bootstrap rules, legal
next-action masks, checkpoint continuation, policy loading, and the original
policy's exhaustive evaluation. Existing static-gallery payload checks remain.
There is no training-convergence test in the suite.

Fresh training on physical GPU 1 (NVIDIA GeForce RTX 5090), seed 0, used 2,048
worlds and 12,288,000 transitions per method. Each run used alpha 0.03, gamma
0.97, and epsilon linearly decreasing from 1.0 to 0.02 over that budget.

.. list-table::
   :header-rows: 1

   * - Policy
     - Collection
     - Mean episode steps
     - Train + evaluation seconds
   * - Original saved table
     - 100%
     - 5.1296
     - Not retrained
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

All evaluations enumerate 3,750 start/target cases; the fresh policies had zero
wrong-target outcomes or truncations. Times are observed wall times for these
short local runs, including compilation and output, not a controlled throughput
benchmark. The new update semantics differ from the historical trainer.
Single-seed measurements do not establish an algorithm ranking.

No matched CPU training benchmark was run, so these timings do not establish a
GPU speedup. CPU and GPU unit-test timings measure a different workload and
should not be used to infer training performance.

The separate custom coordination example also achieved full coordination across
256 evaluation episodes on CPU, using shared parameters and a 4,096-transition
training run. This checks the package boundary on a non-grid environment.

The small suite passes on CPU and GPU. The Sphinx HTML build is checked with
warnings treated as errors. No neural or model-based learning results are claimed.

All six API tutorial notebooks have been executed from fresh CPU kernels, with
outputs saved. Their assertions check worked TD targets, the historical policy,
checkpoint continuation, and the custom environment. Documentation builds render
these outputs with notebook execution disabled.
