import jax
import jax.numpy as jnp

from marlax.agents import IndependentQLearning
from marlax.envs import CooperativeGridWorld


def test_q_update_moves_selected_actions_up():
    learner = IndependentQLearning(num_agents=2, num_states=10, num_actions=5, alpha=0.5)
    state = learner.init(jax.random.key(0))

    state = learner.update(
        state,
        state_id=jnp.array([3], dtype=jnp.int32),
        actions=jnp.array([[2, 1]], dtype=jnp.int32),
        rewards=jnp.array([[1.0, 1.0]], dtype=jnp.float32),
        next_state_id=jnp.array([4], dtype=jnp.int32),
        done=jnp.array([False]),
    )

    assert float(state.q[0, 3, 2]) == 0.5
    assert float(state.q[1, 3, 1]) == 0.5


def test_training_scan_runs():
    env = CooperativeGridWorld(size=5, num_agents=2, num_envs=4)
    env_state = env.reset(jax.random.key(0))
    learner = IndependentQLearning(
        num_agents=env.num_agents,
        num_states=env.num_states,
        num_actions=env.num_actions,
        epsilon_steps=8,
    )
    state = learner.init(jax.random.key(1))

    (_, state), stats = learner.train(env, env_state, state, num_steps=8)

    assert stats.reward.shape == (8,)
    assert int(state.steps) == 8
    assert bool(jnp.all(jnp.isfinite(state.q)))


def test_training_scan_jits():
    env = CooperativeGridWorld(size=5, num_agents=2, num_envs=4)
    env_state = env.reset(jax.random.key(0))
    learner = IndependentQLearning(
        num_agents=env.num_agents,
        num_states=env.num_states,
        num_actions=env.num_actions,
        epsilon_steps=8,
    )
    state = learner.init(jax.random.key(1))

    @jax.jit
    def run(env_state, state):
        return learner.train(env, env_state, state, 8)

    (_, state), stats = run(env_state, state)

    assert stats.reward.shape == (8,)
    assert int(state.steps) == 8
