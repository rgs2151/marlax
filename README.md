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
conda run -n marlax uv pip install --python /home/dev/miniconda3/envs/marlax/bin/python -e ".[gpu,dev,docs,storage]"
```

## Checks

```bash
conda run -n marlax python -m pytest -q
conda run -n marlax make -C docs html
```
