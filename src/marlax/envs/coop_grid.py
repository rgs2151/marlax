from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


MOVES = jnp.array([
    [0, 0],
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
], dtype=jnp.int32)


class GridState(NamedTuple):
    positions: jax.Array
    target: jax.Array
    active: jax.Array
    steps: jax.Array


class GridStep(NamedTuple):
    state: GridState
    state_id: jax.Array
    next_state_id: jax.Array
    rewards: jax.Array
    done: jax.Array
    collected: jax.Array
    activated: jax.Array


def default_zones(size):
    center = size // 2
    return jnp.array([
        [center, size - 1],
        [size - 1, center],
        [center, 0],
        [0, center],
    ], dtype=jnp.int32)


def default_targets(size):
    zones = default_zones(size)
    pairs = jnp.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 3],
        [0, 2],
        [1, 3],
    ], dtype=jnp.int32)
    return zones[pairs]


@dataclass(frozen=True)
class CooperativeGridWorld:
    size: int = 7
    num_agents: int = 2
    num_envs: int = 1
    max_steps: int = 64
    target_reward: float = 1.0
    activation_reward: float = 0.15
    step_reward: float = -0.01
    wrong_target_reward: float = -1.0
    split_target_reward: float = -0.5
    targets: object = None

    def __post_init__(self):
        targets = self.targets
        if targets is None:
            targets = default_targets(self.size)
        object.__setattr__(self, "targets", jnp.asarray(targets, dtype=jnp.int32))
        object.__setattr__(self, "zones", default_zones(self.size))

    @property
    def num_actions(self):
        return MOVES.shape[0]

    @property
    def num_targets(self):
        return self.targets.shape[0]

    @property
    def num_states(self):
        return (self.num_targets + 1) * (self.size * self.size) ** self.num_agents

    @property
    def center(self):
        return jnp.array([self.size // 2, self.size // 2], dtype=jnp.int32)

    def reset(self, key):
        pos_key, target_key = jax.random.split(key)
        positions = jax.random.randint(
            pos_key,
            (self.num_envs, self.num_agents, 2),
            0,
            self.size,
            dtype=jnp.int32,
        )
        target = jax.random.randint(
            target_key,
            (self.num_envs,),
            0,
            self.num_targets,
            dtype=jnp.int32,
        )
        return GridState(
            positions=positions,
            target=target,
            active=jnp.zeros((self.num_envs,), dtype=bool),
            steps=jnp.zeros((self.num_envs,), dtype=jnp.int32),
        )

    def state_id(self, state):
        cell_ids = state.positions[..., 1] * self.size + state.positions[..., 0]
        powers = (self.size * self.size) ** jnp.arange(self.num_agents, dtype=jnp.int32)
        position_id = jnp.sum(cell_ids * powers, axis=-1)
        target_code = jnp.where(state.active, state.target + 1, 0)
        return target_code * (self.size * self.size) ** self.num_agents + position_id

    def step(self, state, actions, key):
        state_id = self.state_id(state)
        positions = jnp.clip(state.positions + MOVES[actions], 0, self.size - 1)

        center_hit = jnp.all(jnp.all(positions == self.center, axis=-1), axis=-1)
        activated = jnp.logical_and(~state.active, center_hit)
        active = jnp.logical_or(state.active, activated)

        target_cells = self.targets[state.target]
        at_target = jnp.all(positions[:, :, None, :] == target_cells[:, None, :, :], axis=-1)
        all_at_first = jnp.all(at_target[:, :, 0], axis=-1)
        all_at_second = jnp.all(at_target[:, :, 1], axis=-1)
        collected = jnp.logical_and(active, jnp.logical_or(all_at_first, all_at_second))

        any_at_first = jnp.any(at_target[:, :, 0], axis=-1)
        any_at_second = jnp.any(at_target[:, :, 1], axis=-1)
        split_target = active & any_at_first & any_at_second & ~collected

        at_zone = jnp.any(jnp.all(positions[:, :, None, :] == self.zones[None, None, :, :], axis=-1), axis=-1)
        at_selected = jnp.any(at_target, axis=-1)
        wrong_target = active & jnp.any(at_zone & ~at_selected, axis=-1)

        steps = state.steps + 1
        timeout = steps >= self.max_steps
        done = collected | wrong_target | timeout

        reward = (
            self.step_reward
            + self.activation_reward * activated.astype(jnp.float32)
            + self.target_reward * collected.astype(jnp.float32)
            + self.wrong_target_reward * wrong_target.astype(jnp.float32)
            + self.split_target_reward * split_target.astype(jnp.float32)
        )
        rewards = jnp.repeat(reward[:, None], self.num_agents, axis=1)

        next_state = GridState(
            positions=positions,
            target=state.target,
            active=active,
            steps=steps,
        )
        reset_state = self.reset(key)
        next_state = GridState(
            positions=jnp.where(done[:, None, None], reset_state.positions, next_state.positions),
            target=jnp.where(done, reset_state.target, next_state.target),
            active=jnp.where(done, reset_state.active, next_state.active),
            steps=jnp.where(done, reset_state.steps, next_state.steps),
        )

        return GridStep(
            state=next_state,
            state_id=state_id,
            next_state_id=self.state_id(next_state),
            rewards=rewards,
            done=done,
            collected=collected,
            activated=activated,
        )
