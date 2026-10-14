from dataclasses import dataclass

import jax
import jax.numpy as jnp

from marlax.core import Decision
from marlax.models import QTable


def action_probabilities(values, mask, epsilon):
    """Epsilon-greedy probabilities with lowest-index greedy tie breaking."""
    greedy = jnp.argmax(jnp.where(mask, values, -jnp.inf), axis=-1)
    uniform = mask / jnp.sum(mask, axis=-1, keepdims=True)
    return epsilon * uniform + (1 - epsilon) * jax.nn.one_hot(greedy, values.shape[-1])


@dataclass(frozen=True)
class Agent:
    """Execute a discrete Q policy with shared or independent parameters."""

    model: object = QTable()
    sharing: str = "independent"

    def slots(self, num_agents):
        return jnp.arange(num_agents) if self.sharing == "independent" else jnp.zeros(num_agents, dtype=jnp.int32)

    def init(self, key, spec):
        count = spec.num_agents if self.sharing == "independent" else 1
        return jax.vmap(lambda k: self.model.init(k, spec.num_observations, spec.num_actions))(
            jax.random.split(key, count)
        )

    def initial_carry(self):
        return self.model.initial_carry()

    def values(self, variables, policy_input):
        slots = self.slots(policy_input.obs.shape[0])
        return jax.vmap(lambda slot, obs: self.model.apply(variables[slot], obs, (), None)[0])(
            slots, policy_input.obs
        )

    def act(self, variables, policy_input, carry, key, exploration=0.0):
        values = self.values(variables, policy_input)
        probabilities = action_probabilities(values, policy_input.action_mask, exploration)
        keys = jax.random.split(key, values.shape[0])
        actions = jax.vmap(lambda k, p: jax.random.categorical(k, jnp.log(p)))(keys, probabilities)
        return Decision(actions.astype(jnp.int32), carry, {"q": values})


@dataclass(frozen=True)
class FrozenPolicy:
    """Greedy inference variables without learning or exploration state."""

    agent: Agent
    variables: object

    def initial_carry(self):
        return self.agent.initial_carry()

    def act(self, policy_input, carry, key):
        values = self.agent.values(self.variables, policy_input)
        actions = jnp.argmax(jnp.where(policy_input.action_mask, values, -jnp.inf), axis=-1)
        return Decision(actions.astype(jnp.int32), carry, {"q": values})
