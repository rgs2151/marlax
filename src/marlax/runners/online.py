from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from marlax.core import Transition
from marlax.methods import SARSA


class TrainState(NamedTuple):
    env_state: object
    timestep: object
    method_state: object
    key: jax.Array
    actions: jax.Array
    episodes: jax.Array


@dataclass(frozen=True)
class Run:
    """Final inference policy, resumable state, and chunk metrics."""

    policy: object
    state: TrainState
    history: list
    output: object


def select_worlds(mask, when_true, when_false):
    return jax.tree.map(
        lambda x, y: jnp.where(mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim)), x, y),
        when_true, when_false,
    )


def sample_actions(agent, variables, policy_input, key, epsilon):
    keys = jax.random.split(key, policy_input.obs.shape[0])
    return jax.vmap(lambda obs, k: agent.act(variables, obs, (), k, epsilon).actions)(policy_input, keys)


def initialize(env, params, agent, method, seed=0, num_envs=64):
    """Initialize batched worlds, learning state, and first behavior actions."""
    key, reset_key, init_key, action_key = jax.random.split(jax.random.PRNGKey(seed), 4)
    states, timesteps = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(reset_key, num_envs), params)
    learning = method.init(init_key, agent, env.spec)
    actions = sample_actions(agent, learning.policy_state, timesteps.policy_input, action_key, method.epsilon(0))
    return TrainState(states, timesteps, learning, key, actions, jnp.array(0))


def run_steps(env, params, method, state, num_steps):
    """Collect and learn through a scan; callers may JIT the whole chunk."""
    num_envs = state.actions.shape[0]
    agent = state.method_state.agent

    def step(carry, _):
        key, step_key, next_key, update_key, reset_key, action_key = jax.random.split(carry.key, 6)
        next_state, timestep = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(step_key, num_envs), carry.env_state, carry.actions, params,
        )
        epsilon = method.epsilon(carry.method_state.env_steps + num_envs)
        next_actions = sample_actions(agent, carry.method_state.policy_state, timestep.policy_input, next_key, epsilon)
        transition = Transition(
            carry.timestep.policy_input, carry.actions, timestep.reward, timestep.policy_input,
            timestep.terminated, timestep.truncated, next_actions,
        )
        learning, metrics = method.update(carry.method_state, transition, update_key)
        done = timestep.terminated | timestep.truncated
        reset_state, reset_timestep = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(reset_key, num_envs), params,
        )
        world = select_worlds(done, reset_state, next_state)
        current = select_worlds(done, reset_timestep, timestep)
        actions = sample_actions(agent, learning.policy_state, current.policy_input, action_key, epsilon)
        if isinstance(method, SARSA):
            actions = jnp.where(done[:, None], actions, next_actions)
        stats = {
            **metrics,
            "reward_per_step": jnp.mean(timestep.reward),
            "episodes": jnp.sum(done),
            "epsilon": epsilon,
        }
        return TrainState(world, current, learning, key, actions, carry.episodes + jnp.sum(done)), stats

    return jax.lax.scan(step, state, None, length=num_steps)


def train(env, params, agent, method, *, seed=0, num_envs=64, env_steps=64_000,
          chunk_steps=1000, output=None, state=None, config=None):
    """Train for an additional world-transition budget, optionally saving a run."""
    from marlax.storage import save_checkpoint, save_policy, runtime_metadata

    if state is None:
        state = initialize(env, params, agent, method, seed, num_envs)
    num_envs = state.actions.shape[0]
    steps = math.ceil(env_steps / num_envs)
    history = []
    advance = jax.jit(lambda s: run_steps(env, params, method, s, chunk_steps))
    directory = Path(output) if output is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
        resolved = {
            "experiment": config,
            "environment_spec": asdict(env.spec),
            "agent": asdict(state.method_state.agent),
            "method": {"name": type(method).__name__, **asdict(method)},
            "seed": seed, "num_envs": num_envs, "requested_additional_env_steps": env_steps,
            "chunk_steps": chunk_steps, "initial_env_steps": int(state.method_state.env_steps),
        }
        (directory / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
        (directory / "provenance.json").write_text(json.dumps(runtime_metadata(), indent=2) + "\n")
    for offset in range(0, steps, chunk_steps):
        count = min(chunk_steps, steps - offset)
        if count == chunk_steps:
            state, stats = advance(state)
        else:
            state, stats = jax.jit(lambda s: run_steps(env, params, method, s, count))(state)
        item = {name: float(np.mean(value)) for name, value in jax.device_get(stats).items()}
        item["episodes"] = int(state.episodes)
        item["env_steps"] = int(state.method_state.env_steps)
        item["updates"] = int(state.method_state.updates)
        history.append(item)
        if directory is not None:
            with (directory / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(item) + "\n")
    policy = method.freeze(state.method_state)
    if directory is not None:
        save_checkpoint(directory / "checkpoint", state)
        save_policy(directory / "policy", policy, env.spec)
    return Run(policy, state, history, directory)
