import jax
import jax.numpy as jnp

from marlax.envs import CooperativeGridWorld, GridState


def test_reset_shapes_and_state_ids():
    env = CooperativeGridWorld(size=5, num_agents=2, num_envs=8)
    state = env.reset(jax.random.key(0))

    assert state.positions.shape == (8, 2, 2)
    assert state.target.shape == (8,)
    assert state.active.shape == (8,)
    assert state.steps.shape == (8,)
    assert jnp.all(env.state_id(state) < env.num_states)


def test_collecting_active_target_finishes_episode():
    env = CooperativeGridWorld(size=5, num_agents=2, num_envs=1, step_reward=0.0)
    target_cell = env.targets[0, 0]
    state = GridState(
        positions=jnp.array([[target_cell, target_cell]], dtype=jnp.int32),
        target=jnp.array([0], dtype=jnp.int32),
        active=jnp.array([True]),
        steps=jnp.array([3], dtype=jnp.int32),
    )

    step = env.step(state, jnp.array([[0, 0]], dtype=jnp.int32), jax.random.key(1))

    assert bool(step.collected[0])
    assert bool(step.done[0])
    assert float(step.rewards[0, 0]) == 1.0


def test_activation_requires_all_agents_at_center():
    env = CooperativeGridWorld(size=5, num_agents=2, num_envs=1, step_reward=0.0)
    center = env.center
    state = GridState(
        positions=jnp.array([[[center[0], center[1]], [0, 0]]], dtype=jnp.int32),
        target=jnp.array([0], dtype=jnp.int32),
        active=jnp.array([False]),
        steps=jnp.array([0], dtype=jnp.int32),
    )

    step = env.step(state, jnp.array([[0, 0]], dtype=jnp.int32), jax.random.key(2))

    assert not bool(step.activated[0])
    assert not bool(step.state.active[0])


def test_activation_rewards_joint_center():
    env = CooperativeGridWorld(size=5, num_agents=2, num_envs=1, step_reward=0.0)
    center = env.center
    state = GridState(
        positions=jnp.array([[center, center]], dtype=jnp.int32),
        target=jnp.array([0], dtype=jnp.int32),
        active=jnp.array([False]),
        steps=jnp.array([0], dtype=jnp.int32),
    )

    step = env.step(state, jnp.array([[0, 0]], dtype=jnp.int32), jax.random.key(3))

    assert bool(step.activated[0])
    assert bool(step.state.active[0])
    assert bool(jnp.isclose(step.rewards[0, 0], 0.15))
