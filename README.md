# MARLAX

JAX-first cooperative multi-agent reinforcement learning.

The first target is a small, fast tabular Q-learning stack:

- batched cooperative gridworld environments
- independent Q-learning agents with shared team rewards
- JAX scan-friendly training loops

The longer-term direction is a method zoo and environment zoo for cooperative MARL.

## Environment

```bash
conda env create -f environment.yml
conda run -n marlax uv pip install --python /home/dev/miniconda3/envs/marlax/bin/python -e ".[gpu,dev,storage,viz]"
```

## Checks

```bash
conda run -n marlax python -m pytest -q
conda run -n marlax python experiments/coop_grid_q_learning.py
```

## Gallery

```bash
python -m http.server 8000 --directory site
```
