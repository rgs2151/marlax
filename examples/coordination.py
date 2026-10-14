from dataclasses import dataclass
import json

import jax.numpy as jnp

import marlax as mx
from marlax.methods import ExpectedSARSA
from marlax.models import QTable


@dataclass(frozen=True)
class Coordination:
    """A one-step cooperative game rewarding the joint action (1, 1)."""

    @property
    def spec(self):
        return mx.EnvSpec(2, 1, 2, "coordination_v1")

    def default_params(self):
        return jnp.array(1.0)

    def observe(self, state, params):
        return mx.PolicyInput(jnp.zeros(2, dtype=jnp.int32), jnp.ones((2, 2), dtype=bool))

    def global_state(self, state, params):
        return jnp.zeros(1)

    def reset(self, key, params):
        state = jnp.array(0)
        return state, mx.TimeStep(
            self.observe(state, params), self.global_state(state, params),
            jnp.zeros(2), jnp.array(False), jnp.array(False), {"coordinated": jnp.array(False)},
        )

    def step(self, key, state, actions, params):
        coordinated = jnp.all(actions == 1)
        state = state + 1
        return state, mx.TimeStep(
            self.observe(state, params), self.global_state(state, params),
            jnp.full(2, params * coordinated), jnp.array(True), jnp.array(False),
            {"coordinated": coordinated},
        )


if __name__ == "__main__":
    env = Coordination()
    params = env.default_params()
    run = mx.train(env, params, mx.Agent(QTable(), sharing="shared"),
                   ExpectedSARSA(alpha=0.2, epsilon_steps=4096),
                   num_envs=32, env_steps=4096, chunk_steps=128)
    print(json.dumps(mx.evaluate(run.policy, env, params, max_steps=1), indent=2))
