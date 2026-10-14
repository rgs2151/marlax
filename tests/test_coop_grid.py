import jax
import jax.numpy as jnp
import numpy as np

import marlax as mx
from marlax.envs import CooperativeGrid, GridState
from marlax.models import QTable


def test_grid_activation_collection_and_raw_terminal_state():
    env = CooperativeGrid()
    params = env.default_params()
    state = GridState(jnp.array([[2, 2], [2, 1]]), jnp.array(0), jnp.array(False), jnp.array(0))
    _, before = env.step(jax.random.PRNGKey(0), state, jnp.array([0, 0]), params)
    assert not before.metrics["activated"]
    state, activated = env.step(jax.random.PRNGKey(0), state, jnp.array([0, 2]), params)
    assert activated.metrics["activated"]
    assert jnp.isclose(activated.reward[0], 0.19)
    state, _ = env.step(jax.random.PRNGKey(0), state, jnp.array([2, 2]), params)
    state, collected = env.step(jax.random.PRNGKey(0), state, jnp.array([2, 2]), params)
    assert collected.terminated
    assert jnp.array_equal(state.positions, jnp.array([[2, 4], [2, 4]]))
    assert state.active


def test_wrong_split_and_timeout_are_distinct():
    env = CooperativeGrid()
    params = env.default_params()._replace(time_limit=1)
    state = GridState(jnp.array([[2, 4], [4, 2]]), jnp.array(0), jnp.array(True), jnp.array(0))
    _, step = env.step(jax.random.PRNGKey(0), state, jnp.array([0, 0]), params)
    assert step.truncated and not step.terminated
    assert jnp.isclose(step.reward[0], -0.41)
    state = state._replace(positions=jnp.array([[2, 0], [4, 2]]))
    _, step = env.step(jax.random.PRNGKey(0), state, jnp.array([0, 0]), params)
    assert step.terminated and step.metrics["wrong_target"]


def test_saved_policy_on_all_original_start_target_cases():
    env = CooperativeGrid()
    with np.load("stores/coop_grid_q_learning/latest/q_values.npz", allow_pickle=False) as saved:
        policy = mx.FrozenPolicy(mx.Agent(QTable()), jnp.asarray(saved["q"]))
    report = mx.evaluate(policy, env, env.default_params(), initial_states=env.all_starts())
    assert report["cases"] == 3750
    assert report["collected_rate"] == 1.0
    assert np.isclose(report["mean_episode_steps"], 5.1296)
