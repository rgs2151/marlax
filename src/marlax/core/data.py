from dataclasses import dataclass
from typing import NamedTuple

import jax


@dataclass(frozen=True)
class EnvSpec:
    """Static finite-observation, discrete-action environment specification."""

    num_agents: int
    num_observations: int
    num_actions: int
    observation_schema: str


class PolicyInput(NamedTuple):
    obs: jax.Array
    action_mask: jax.Array


class TimeStep(NamedTuple):
    policy_input: PolicyInput
    global_state: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    metrics: dict


class Transition(NamedTuple):
    policy_input: PolicyInput
    actions: jax.Array
    reward: jax.Array
    next_policy_input: PolicyInput
    terminated: jax.Array
    truncated: jax.Array
    next_actions: jax.Array


class Decision(NamedTuple):
    actions: jax.Array
    carry: tuple
    extras: dict
