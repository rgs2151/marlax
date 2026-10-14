# MARLAX

[Documentation](https://rgs2151.github.io/marlax/) ·
[API tutorials](https://rgs2151.github.io/marlax/tutorials.html) ·
[Tool inventory](https://rgs2151.github.io/marlax/inventory.html) ·
[Interactive gallery](https://rgs2151.github.io/marlax/gallery/)

[![Deployment](https://github.com/rgs2151/marlax/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/rgs2151/marlax/actions/workflows/pages.yml)
[![Documentation](https://img.shields.io/badge/docs-Sphinx-blue)](https://rgs2151.github.io/marlax/)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](.github/workflows/pages.yml)
[![JAX](https://img.shields.io/badge/backend-JAX-009688)](https://github.com/jax-ml/jax)
[![Status](https://img.shields.io/badge/status-alpha-orange)](pyproject.toml)

MARLAX is a JAX library for cooperative multi-agent reinforcement learning.
Compose an environment, a model, an agent, and a learning method; train across
batched worlds; then save a frozen policy for forward simulation.

Version **0.1.0 (alpha)**. Train tabular policies with Q-learning, SARSA, or
Expected SARSA. Start with the cooperative grid or define a custom environment
using the environment protocol.

## Install

From a checkout:

```bash
git clone https://github.com/rgs2151/marlax.git
cd marlax
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,docs]'
```

Documentation builds use Python 3.13. For local GPU training, add the `gpu` extra for CUDA 13.
The `tutorials` extra adds Jupyter and plotting; `viz` adds matplotlib alone.
The optional conda environment is defined in [environment.yml](environment.yml).

## Train and evaluate

```python
import marlax as mx
from marlax.envs import CooperativeGrid
from marlax.methods import QLearning
from marlax.models import QTable

env = CooperativeGrid(size=5, num_agents=2)
params = env.default_params()
agent = mx.Agent(QTable(), sharing="independent")
method = QLearning(epsilon_steps=64_000)

run = mx.train(
    env, params, agent, method,
    seed=0, num_envs=64, env_steps=64_000,
)
report = mx.evaluate(
    run.policy, env, params,
    initial_states=env.all_starts(), max_steps=params.time_limit,
)
```

This small budget demonstrates the API. Use the demo recipe for a longer learning
run. Select `SARSA` or `ExpectedSARSA` to change the learning rule. Environment
state and model variables are explicit; batching and resets belong to the runner.

## What's available

| Component | Implemented tools |
| --- | --- |
| Environment | Cooperative grid, explicit scenarios, exhaustive start-state enumeration |
| Model and agents | Q-table, optimistic initialization, shared or independent parameters, action masks |
| Learning | Q-learning, SARSA, Expected SARSA with averaged batched TD updates |
| Execution | JAX `vmap`/`scan` training, greedy frozen policies, common training/evaluation dynamics |
| Storage | Compressed NPZ arrays, JSON manifests, full checkpoints including RNG and pending actions |
| Experiments | Training, evaluation, resume, pretrained-policy evaluation, and SVG diagnostics |
| Documentation | User guide, six executed notebooks, tool inventory, and generated API reference |

See the [full inventory](https://rgs2151.github.io/marlax/inventory.html) for
individual functions and supported combinations.

## Tutorials

| Tutorial | Learn to |
| --- | --- |
| [Environment and timestep](https://rgs2151.github.io/marlax/notebooks/01_environment.html) | Reset, step, observe, terminate, and batch worlds |
| [Models and agents](https://rgs2151.github.io/marlax/notebooks/02_models_agents.html) | Initialize parameters, share tables, mask actions, and freeze policies |
| [Learning methods](https://rgs2151.github.io/marlax/notebooks/03_learning_methods.html) | Inspect TD targets and select among the three methods |
| [Training and evaluation](https://rgs2151.github.io/marlax/notebooks/04_training_evaluation.html) | Control JIT chunks, evaluate the saved policy, and plot a rollout |
| [Checkpoints](https://rgs2151.github.io/marlax/notebooks/05_checkpoints.html) | Save inference models and resume training exactly on the same setup |
| [Custom environments](https://rgs2151.github.io/marlax/notebooks/06_custom_environment.html) | Build a cooperative game and reuse the existing learning stack |

The published pages include saved outputs and notebook downloads. Source notebooks
live in [docs/source/notebooks](docs/source/notebooks) and run on CPU.

## Experiment recipe

```bash
python experiments/coop_grid_td/run.py train --method q_learning --profile smoke
python experiments/coop_grid_td/run.py train --method expected_sarsa --profile demo
python experiments/coop_grid_td/run.py evaluate --run runs/coop_grid_td/RUN_ID
python experiments/coop_grid_td/run.py resume --run runs/coop_grid_td/RUN_ID --env-steps 2048000
python experiments/coop_grid_td/run.py evaluate-saved
```

Runs save configuration, source snapshots, metrics, checkpoints, inference policies,
and evaluation under fresh `runs/` directories. See
[the recipe](experiments/coop_grid_td/README.md)
and [storage guide](https://rgs2151.github.io/marlax/storage.html).

Example runs reached **100% collection across 3,750 start/target cases** for all
three methods after 12.288 million world transitions using seed 0. See
[evaluation results](https://rgs2151.github.io/marlax/validation.html) for the
configuration and measurements.

## CI and publishing

[GitHub Actions](.github/workflows/pages.yml) builds Sphinx from saved notebook
outputs and publishes successful builds on `main` to GitHub Pages:

- `/marlax/` — documentation, tutorials, inventory, and API reference.
- `/marlax/gallery/` — interactive gallery.

Pull requests produce a downloadable `marlax-site` artifact. The workflow also
supports manual dispatch. Tests and documentation checks run locally.

For local verification:

```bash
JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python -m sphinx -W --keep-going -b html docs/source docs/build/html
python scripts/check_docs.py docs/build/html
```

Use [STYLE.md](STYLE.md) when contributing diagnostic plots.
