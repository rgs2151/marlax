import jax
import jax.numpy as jnp
import numpy as np

from marlax.runners.online import select_worlds


def evaluate(policy, env, params, *, initial_states=None, num_episodes=256, seed=0, max_steps=48):
    """Evaluate frozen behavior through the same environment step used for learning."""
    key = jax.random.PRNGKey(seed)
    if initial_states is None:
        key, reset_key = jax.random.split(key)
        initial_states, _ = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(reset_key, num_episodes), params,
        )
    count = jax.tree.leaves(initial_states)[0].shape[0]

    def rollout(states):
        def step(carry, _):
            state, key, done = carry
            key, act_key, step_key = jax.random.split(key, 3)
            inputs = jax.vmap(env.observe, in_axes=(0, None))(state, params)
            actions = jax.vmap(lambda obs, k: policy.act(obs, (), k).actions)(
                inputs, jax.random.split(act_key, count),
            )
            next_state, timestep = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(step_key, count), state, actions, params,
            )
            ended = timestep.terminated | timestep.truncated
            metrics = {name: value * ~done for name, value in timestep.metrics.items()}
            record = {
                "reward": jnp.mean(timestep.reward, axis=-1) * ~done,
                "live": ~done, "terminated": ~done & timestep.terminated,
                "truncated": ~done & timestep.truncated & ~timestep.terminated,
                **metrics,
            }
            return (select_worlds(done, state, next_state), key, done | ended), record

        return jax.lax.scan(step, (states, key, jnp.zeros(count, dtype=bool)), None, length=max_steps)

    (_, _, done), records = jax.jit(rollout)(initial_states)
    records = jax.device_get(records)
    summary = {
        "cases": count,
        "mean_return": float(records["reward"].sum(axis=0).mean()),
        "mean_episode_steps": float(records["live"].sum(axis=0).mean()),
        "termination_rate": float(records["terminated"].any(axis=0).mean()),
        "truncation_rate": float(records["truncated"].any(axis=0).mean()),
        "unfinished_rate": float(np.mean(~np.asarray(done))),
    }
    for name in records.keys() - {"reward", "live", "terminated", "truncated"}:
        summary[name + "_rate"] = float(records[name].any(axis=0).mean())
    return summary
