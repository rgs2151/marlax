import jax
import numpy as np
from tqdm import tqdm

from config import CHUNK_STEPS, CHUNKS_PER_ITERATION, ITERATIONS, MAX_STEPS, NUM_ENVS, SIZE
from evaluate import evaluate_policy
from marlax.agents import IndependentQLearning
from marlax.envs import CooperativeGridWorld


def make_env():
    return CooperativeGridWorld(
        size=SIZE,
        num_agents=2,
        num_envs=NUM_ENVS,
        max_steps=MAX_STEPS,
        target_reward=1.0,
        activation_reward=0.2,
        step_reward=-0.01,
        wrong_target_reward=-1.0,
        split_target_reward=-0.4,
    )


def make_learner(env):
    return IndependentQLearning(
        num_agents=env.num_agents,
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=0.03,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_steps=ITERATIONS * CHUNKS_PER_ITERATION * CHUNK_STEPS,
    )


def train_policy():
    env = make_env()
    learner = make_learner(env)
    env_state = env.reset(jax.random.key(0))
    learner_state = learner.init(jax.random.key(1))

    @jax.jit
    def run_chunk(env_state, learner_state):
        return learner.train(env, env_state, learner_state, CHUNK_STEPS)

    history = []
    total_chunks = ITERATIONS * CHUNKS_PER_ITERATION
    with tqdm(total=total_chunks, desc="Training Q values", unit="chunk") as progress:
        for iteration in range(ITERATIONS):
            chunk_rewards = []
            chunk_done = []
            chunk_collected = []
            chunk_epsilon = []
            for _ in range(CHUNKS_PER_ITERATION):
                (env_state, learner_state), stats = run_chunk(env_state, learner_state)
                stats = jax.tree.map(lambda value: np.asarray(value), stats)
                chunk_rewards.append(float(np.mean(stats.reward)))
                chunk_done.append(float(np.mean(stats.done)))
                chunk_collected.append(float(np.mean(stats.collected)))
                chunk_epsilon.append(float(np.mean(stats.epsilon)))
                progress.update(1)

            q = np.asarray(learner_state.q)
            eval_stats = evaluate_policy(q, env)
            history.append({
                "iteration": iteration + 1,
                "train_reward": float(np.mean(chunk_rewards)),
                "train_done": float(np.mean(chunk_done)),
                "train_collected": float(np.mean(chunk_collected)),
                "epsilon": float(np.mean(chunk_epsilon)),
                "eval_collection_rate": eval_stats["collection_rate"],
                "eval_activation_rate": eval_stats["activation_rate"],
                "eval_wrong_target_rate": eval_stats["wrong_target_rate"],
                "eval_timeout_rate": eval_stats["timeout_rate"],
                "eval_mean_done_steps": eval_stats["mean_done_steps"],
                "eval_mean_collection_steps": eval_stats["mean_collection_steps"],
                "eval_visited_states": eval_stats["visited_states"],
                "eval_mean_q_margin": eval_stats["mean_q_margin"],
                "eval_median_q_margin": eval_stats["median_q_margin"],
            })

    return env, np.asarray(learner_state.q), history
