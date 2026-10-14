from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class QTable:
    """A finite observation-to-action-value table for one parameter slot."""

    initial_value: float = 0.0

    def init(self, key, input_spec, output_spec):
        return jnp.full((input_spec, output_spec), self.initial_value, dtype=jnp.float32)

    def initial_carry(self):
        return ()

    def apply(self, variables, inputs, carry, key):
        return variables[inputs], carry
