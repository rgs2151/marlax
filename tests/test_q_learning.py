from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import marlax as mx
from marlax.envs import CooperativeGrid
from marlax.methods import ExpectedSARSA, QLearning, SARSA
from marlax.models import QTable


@pytest.mark.parametrize("method_class,expected", [(QLearning, 1.5), (SARSA, 1.0), (ExpectedSARSA, 1.45)])
def test_td_targets_average_duplicates_and_bootstrap_truncations(method_class, expected):
    agent = mx.Agent(QTable(), sharing="shared")
    method = method_class(alpha=0.5, gamma=0.5, epsilon_start=0.2, epsilon_end=0.2)
    state = method.init(jax.random.PRNGKey(0), agent, mx.EnvSpec(2, 4, 3, "test"))
    state = replace(state, policy_state=state.policy_state.at[0, 1].set(jnp.array([2., 4., 100.])))
    current = mx.PolicyInput(jnp.zeros((4, 2), dtype=jnp.int32), jnp.ones((4, 2, 3), dtype=bool))
    following = mx.PolicyInput(jnp.ones((4, 2), dtype=jnp.int32), jnp.tile(jnp.array([True, True, False]), (4, 2, 1)))
    batch = mx.Transition(current, jnp.zeros((4, 2), dtype=jnp.int32), jnp.ones((4, 2)), following,
                          jnp.zeros(4, dtype=bool), jnp.ones(4, dtype=bool), jnp.zeros((4, 2), dtype=jnp.int32))
    updated, _ = method.update(state, batch, jax.random.PRNGKey(1))
    assert jnp.isclose(updated.policy_state[0, 0, 0], expected)
    terminal, _ = method.update(state, batch._replace(terminated=jnp.ones(4, dtype=bool)), jax.random.PRNGKey(1))
    assert terminal.policy_state[0, 0, 0] == 0.5
    np.testing.assert_array_equal(updated.policy_state[0, 1], [2., 4., 100.])


@pytest.mark.parametrize("method_class", [QLearning, SARSA, ExpectedSARSA])
def test_scan_checkpoint_resume_and_policy_roundtrip(tmp_path, method_class):
    env = CooperativeGrid(size=3)
    params = env.default_params()._replace(time_limit=2)
    agent, method = mx.Agent(QTable()), method_class(epsilon_steps=32)
    initial = mx.initialize(env, params, agent, method, num_envs=4)
    advance = jax.jit(lambda state: mx.run_steps(env, params, method, state, 4)[0])
    middle = advance(initial)
    mx.save_checkpoint(tmp_path / "checkpoint", middle)
    restored = mx.load_checkpoint(tmp_path / "checkpoint", initial)
    expected, resumed = advance(middle), advance(restored)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a, b)
    assert int(resumed.method_state.env_steps) == 32
    assert int(resumed.episodes) >= 16
    assert np.isfinite(resumed.method_state.policy_state).all()
    policy = method.freeze(resumed.method_state)
    mx.save_policy(tmp_path / "policy", policy, env.spec)
    loaded = mx.load_policy(tmp_path / "policy")
    _, timestep = env.reset(jax.random.PRNGKey(9), params)
    np.testing.assert_array_equal(loaded.act(timestep.policy_input, (), None).actions,
                                  policy.act(timestep.policy_input, (), None).actions)
