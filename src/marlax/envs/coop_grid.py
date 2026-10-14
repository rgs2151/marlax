from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from marlax.core import EnvSpec, PolicyInput, TimeStep


MOVES = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))


class GridState(NamedTuple):
    positions: jax.Array
    target: jax.Array
    active: jax.Array
    steps: jax.Array


class GridParams(NamedTuple):
    time_limit: int = 48
    target_reward: float = 1.0
    activation_reward: float = 0.2
    step_reward: float = -0.01
    wrong_target_reward: float = -1.0
    split_target_reward: float = -0.4


@dataclass(frozen=True)
class CooperativeGrid:
    """Animals meet at the center, then collect one jointly selected edge target."""

    size: int = 5
    num_agents: int = 2

    @property
    def zones(self):
        c = self.size // 2
        return jnp.array([[c, self.size - 1], [self.size - 1, c], [c, 0], [0, c]], dtype=jnp.int32)

    @property
    def targets(self):
        return self.zones[jnp.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 2], [1, 3]])]

    @property
    def center(self):
        return jnp.array([self.size // 2, self.size // 2], dtype=jnp.int32)

    @property
    def spec(self):
        return EnvSpec(self.num_agents, 7 * (self.size ** 2) ** self.num_agents, 5, "coop_grid_joint_index_v1")

    def default_params(self):
        return GridParams()

    def state_id(self, state):
        cells = state.positions[:, 1] * self.size + state.positions[:, 0]
        position = jnp.sum(cells * (self.size ** 2) ** jnp.arange(self.num_agents))
        code = jnp.where(state.active, state.target + 1, 0)
        return code * (self.size ** 2) ** self.num_agents + position

    def observe(self, state, params):
        return PolicyInput(
            jnp.full((self.num_agents,), self.state_id(state), dtype=jnp.int32),
            jnp.ones((self.num_agents, 5), dtype=bool),
        )

    def global_state(self, state, params):
        return jnp.concatenate((state.positions.reshape(-1), jnp.array([state.target, state.active, state.steps])))

    def timestep(self, state, params, reward=0.0, terminated=False, truncated=False, activated=False, collected=False, wrong_target=False):
        return TimeStep(
            self.observe(state, params),
            self.global_state(state, params),
            jnp.full((self.num_agents,), reward, dtype=jnp.float32),
            jnp.asarray(terminated),
            jnp.asarray(truncated),
            {"activated": jnp.asarray(activated), "collected": jnp.asarray(collected), "wrong_target": jnp.asarray(wrong_target)},
        )

    def reset(self, key, params):
        pos_key, target_key = jax.random.split(key)
        state = GridState(
            jax.random.randint(pos_key, (self.num_agents, 2), 0, self.size),
            jax.random.randint(target_key, (), 0, 6),
            jnp.array(False),
            jnp.array(0, dtype=jnp.int32),
        )
        return state, self.timestep(state, params)

    def step(self, key, state, actions, params):
        positions = jnp.clip(state.positions + jnp.asarray(MOVES)[actions], 0, self.size - 1)
        activated = ~state.active & jnp.all(positions == self.center)
        active = state.active | activated
        at_target = jnp.all(positions[:, None, :] == self.targets[state.target][None, :, :], axis=-1)
        collected = active & jnp.any(jnp.all(at_target, axis=0))
        split = active & jnp.all(jnp.any(at_target, axis=0)) & ~collected
        at_zone = jnp.any(jnp.all(positions[:, None, :] == self.zones[None, :, :], axis=-1), axis=-1)
        wrong = active & jnp.any(at_zone & ~jnp.any(at_target, axis=-1))
        next_state = GridState(positions, state.target, active, state.steps + 1)
        reward = (params.step_reward + params.activation_reward * activated
                  + params.target_reward * collected + params.wrong_target_reward * wrong
                  + params.split_target_reward * split)
        return next_state, self.timestep(
            next_state, params, reward, collected | wrong, next_state.steps >= params.time_limit,
            activated, collected, wrong,
        )

    def all_starts(self):
        """Enumerate every ordered position tuple and target, initially inactive."""
        indices = jnp.arange((self.size ** 2) ** self.num_agents)
        cells = (indices[:, None] // (self.size ** 2) ** jnp.arange(self.num_agents)) % (self.size ** 2)
        positions = jnp.stack((cells % self.size, cells // self.size), axis=-1)
        count = len(indices)
        return GridState(
            jnp.tile(positions, (6, 1, 1)),
            jnp.repeat(jnp.arange(6), count),
            jnp.zeros(6 * count, dtype=bool),
            jnp.zeros(6 * count, dtype=jnp.int32),
        )
