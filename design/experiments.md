# Experiment workflow

## A recipe is ordinary Python

Each experiment asks a specific question and composes public components. Keep
configuration small and explicit; use a Python dataclass and a few CLI arguments
before considering a larger configuration framework. Resolve the entire recipe
into JSON at run start. Custom components provide a stable name and constructor
configuration; source is identified by the repository revision.

The initial recipe is `experiments/coop_grid_td/run.py`, comparing Q-learning,
SARSA, and Expected SARSA on the same cooperative grid. Environment rules and
learning updates belong in the package; the recipe selects them. Evaluation,
plotting, and export should be callable independently of training.

Proposed commands, available only after implementation:

```bash
python experiments/coop_grid_td/run.py train --method q_learning --seed 0 --profile smoke
python experiments/coop_grid_td/run.py train --method q_learning --seed 0 --profile demo
python experiments/coop_grid_td/run.py evaluate --run runs/coop_grid_td/<run-id>
python experiments/coop_grid_td/run.py plot --run runs/coop_grid_td/<run-id>
python experiments/coop_grid_td/run.py export --run runs/coop_grid_td/<run-id> --demo mice-rendezvous
```

`smoke` is a small execution check, not evidence of convergence. `demo` is a
deliberate learning run. Changing a plot, skin, page layout, or exporter must never
implicitly rerun learning. Imports have no training or file-writing side effects.

## Run layout and ownership

```text
runs/coop_grid_td/<run-id>/
  config.json                  # fully resolved recipe, including seeds
  provenance.json              # commit, dirty status, versions, device details
  metrics.jsonl                # incremental training/evaluation measurements
  checkpoints/
    final/                     # variables, optimizer, schedules, counters, keys
  evaluation.json              # policy/checkpoint identity and scenario set
  plots/                       # derived, follow STYLE.md
```

A run ID is unique and human-readable, for example
`20261014T150000Z-q_learning-seed0`. Use UTC in machine identifiers and metadata.
Outputs live outside the installed Python package. Resume uses the saved resolved
configuration and full state; starting with only frozen weights is a new run.
If the tree is dirty, save the diff with provenance so a commit hash does not
misrepresent the actual training source.

The current `stores/.../latest` data is a useful historical reference. Future runs
do not overwrite it. A small catalog can point to a selected run without renaming
or mutating that run's contents.

## Minimum useful metadata

| Group | Values |
| --- | --- |
| World | Environment ID/version, structure, rewards, observation mode, time-limit meaning |
| Agent | Model architecture, parameter sharing, feature ordering, action selection |
| Method | Algorithm, alpha/optimizer, gamma, exploration schedule, target/planning settings |
| Budget | Parallel worlds, world transitions, actual counters, rollout/update sizes |
| Randomness | Root seed and initialization/training/evaluation stream convention |
| Execution | Git revision/dirty diff, Python/JAX and optional dependency versions, device, timing |
| Evaluation | Checkpoint identity, scenario-set ID, greedy/stochastic mode, metrics with denominators |

Save runtime state and optimizer state for resumption. Save only inference state
in the browser artifact. These are different products with different lifetimes.

## Evaluation

Use the package's own environment step for training and Python evaluation. Remove
the separate NumPy implementation when migrating; it is a second definition of
the rules today. Evaluation receives a frozen policy and does no updates.

For the current 5×5, two-agent task, enumerate all 625 ordered start-position
pairs for each of six target pairs: 3,750 cases. Start each episode inactive and
at step zero. Include the original six showcase scenarios for continuity, but
do not treat them as the entire evaluation population.

Report success fraction per episode, activation fraction per episode, disjoint
terminal outcomes, mean episode length, mean successful length, and team return.
If reward is repeated across agents, use one copy (or their mean), not their sum.
Collection frequency per world step is a different throughput measure and must
not share an unlabeled denominator with episode success.

For new larger worlds, save a fixed evaluation scenario set separate from training
seeds. Evaluate out-of-distribution layouts only when making a generalization
claim. Report a single-seed demo as a demo; use repeated training seeds when
making comparisons about the methods themselves. No need to pay for that study
merely to exercise the API.

## Selecting a public demo

Training writes runs. Evaluation records behavior. Export selects a checkpoint
and writes `artifacts/<demo>/<revision>/`. A gallery catalog chooses the artifact
revision and theme; publishing builds the static site from those files.

Use the final checkpoint initially. If selecting the best checkpoint later,
record the selection metric and evaluation set, and reserve another set for
reported generalization. Do not quietly pick visually pleasing rollouts as proof
that the overall policy works.

## Minimal verification budget

Keep the existing fast checks during migration; adapt them as APIs change. Spend
additional test effort only on the following boundaries as they are implemented:

1. A tiny environment trajectory covering activation, terminal output, and reset
   ownership, with one JIT/batch smoke pass.
2. A hand-computable TD update, including duplicate indices and the terminal
   bootstrap convention.
3. One small exported trajectory checked in Python and JavaScript, including
   state encoding and action choice.

These can be consolidated into a few tests. No GPU requirement, training
convergence tests, screenshots, exhaustive environment/method matrix, or coverage
targets. Run expensive training only for an actual experiment. During this
documentation-only pass, verify document consistency and links rather than rerun
the training or test suite.
