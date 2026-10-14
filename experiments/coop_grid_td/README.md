# Cooperative-grid TD methods

Train Q-learning, SARSA, or Expected SARSA on the same two-agent grid. Both animals
observe their joint position and the target pair after center activation. The
experiment trains a policy, saves a checkpoint, and evaluates every start/target
combination.

```bash
python experiments/coop_grid_td/run.py train --method q_learning --profile smoke
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/coop_grid_td/run.py train --method q_learning --profile demo
python experiments/coop_grid_td/run.py evaluate --run runs/coop_grid_td/RUN_ID
python experiments/coop_grid_td/run.py plot --run runs/coop_grid_td/RUN_ID
python experiments/coop_grid_td/run.py resume --run runs/coop_grid_td/RUN_ID --env-steps 2048000
python experiments/coop_grid_td/run.py evaluate-saved
```

The smoke profile uses 1,024 world transitions; the demo uses 12,288,000. Override
with `--env-steps`. Runs have unique directories and save configuration, numerical
training state, a standalone greedy policy, and exhaustive evaluation over 3,750
start/target cases. `--output` must point to a new directory.

`evaluate-saved` loads the pretrained NumPy table in
`stores/coop_grid_q_learning/latest/` and reports its evaluation metrics.

See the Sphinx guides in `docs/source/` for contracts, storage, extension, and API
reference.
