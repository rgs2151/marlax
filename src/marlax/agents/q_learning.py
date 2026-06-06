from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


class QLearningState(NamedTuple):
    q: jax.Array
    key: jax.Array
    steps: jax.Array


class TrainStats(NamedTuple):
    reward: jax.Array
    done: jax.Array
    collected: jax.Array
    epsilon: jax.Array


@dataclass(frozen=True)
class IndependentQLearning:
    num_agents: int
    num_states: int
    num_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_steps: int = 10_000

    def init(self, key):
        return QLearningState(
            q=jnp.zeros((self.num_agents, self.num_states, self.num_actions), dtype=jnp.float32),
            key=key,
            steps=jnp.array(0, dtype=jnp.int32),
        )

    def epsilon(self, steps):
        frac = jnp.minimum(steps / self.epsilon_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, state, state_id, epsilon):
        key, random_key, explore_key = jax.random.split(state.key, 3)
        values = state.q[:, state_id, :]
        greedy_actions = jnp.argmax(values, axis=-1).T
        random_actions = jax.random.randint(random_key, greedy_actions.shape, 0, self.num_actions)
        explore = jax.random.uniform(explore_key, greedy_actions.shape) < epsilon
        actions = jnp.where(explore, random_actions, greedy_actions).astype(jnp.int32)
        return actions, state._replace(key=key)

    def update(self, state, state_id, actions, rewards, next_state_id, done):
        agent_ids = jnp.arange(self.num_agents)[:, None]
        batch_state_id = state_id[None, :]
        batch_actions = actions.T
        old = state.q[agent_ids, batch_state_id, batch_actions]
        next_value = jnp.max(state.q[:, next_state_id, :], axis=-1)
        target = rewards.T + self.gamma * (1.0 - done[None, :]) * next_value
        delta = self.alpha * (target - old)
        q = state.q.at[agent_ids, batch_state_id, batch_actions].add(delta)
        return state._replace(q=q, steps=state.steps + 1)

    def train(self, env, env_state, state, num_steps):
        def train_step(carry, _):
            env_state, state = carry
            state_id = env.state_id(env_state)
            epsilon = self.epsilon(state.steps)
            actions, state = self.act(state, state_id, epsilon)
            key, step_key = jax.random.split(state.key)
            step = env.step(env_state, actions, step_key)
            state = self.update(state._replace(key=key), step.state_id, actions, step.rewards, step.next_state_id, step.done)
            stats = TrainStats(
                reward=jnp.mean(step.rewards),
                done=jnp.mean(step.done.astype(jnp.float32)),
                collected=jnp.mean(step.collected.astype(jnp.float32)),
                epsilon=epsilon,
            )
            return (step.state, state), stats

        return jax.lax.scan(train_step, (env_state, state), None, length=num_steps)
