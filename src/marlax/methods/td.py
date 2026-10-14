from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from marlax.agents import FrozenPolicy, action_probabilities


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MethodState:
    """Numerical learning state with a static agent description."""

    policy_state: object
    env_steps: object
    updates: object
    agent: object = field(metadata={"static": True})


@dataclass(frozen=True)
class QLearning:
    """Batched tabular Q-learning, averaging TD errors at duplicate indices."""

    alpha: float = 0.03
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_steps: int = 12_288_000

    def init(self, key, agent, spec):
        return MethodState(agent.init(key, spec), jnp.array(0), jnp.array(0), agent)

    def epsilon(self, env_steps):
        fraction = jnp.minimum(env_steps / self.epsilon_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def next_value(self, values, batch, epsilon):
        return jnp.max(jnp.where(batch.next_policy_input.action_mask, values, -jnp.inf), axis=-1)

    def update(self, state, batch, key):
        q = state.policy_state
        slots = state.agent.slots(batch.actions.shape[-1])[None, :]
        indices = (slots, batch.policy_input.obs, batch.actions)
        next_q = q[slots, batch.next_policy_input.obs]
        steps = state.env_steps + batch.actions.shape[0]
        bootstrap = self.next_value(next_q, batch, self.epsilon(steps))
        target = batch.reward + self.gamma * (~batch.terminated[:, None]) * bootstrap
        delta = target - q[indices]
        totals = jnp.zeros_like(q).at[indices].add(delta)
        counts = jnp.zeros_like(q).at[indices].add(1)
        q = q + self.alpha * totals / jnp.maximum(counts, 1)
        return replace(state, policy_state=q, env_steps=steps, updates=state.updates + 1), {
            "mean_abs_td": jnp.mean(jnp.abs(delta)),
        }

    def freeze(self, state):
        return FrozenPolicy(state.agent, state.policy_state)


@dataclass(frozen=True)
class SARSA(QLearning):
    """On-policy tabular TD using the retained next behavior action."""

    def next_value(self, values, batch, epsilon):
        return jnp.take_along_axis(values, batch.next_actions[..., None], axis=-1)[..., 0]


@dataclass(frozen=True)
class ExpectedSARSA(QLearning):
    """Tabular TD under the next epsilon-greedy action distribution."""

    def next_value(self, values, batch, epsilon):
        probabilities = action_probabilities(values, batch.next_policy_input.action_mask, epsilon)
        return jnp.sum(probabilities * values, axis=-1)
