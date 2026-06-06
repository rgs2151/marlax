from pathlib import Path
import json
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from marlax.agents import IndependentQLearning
from marlax.envs import CooperativeGridWorld
from marlax.envs.coop_grid import MOVES


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "coop_grid_q_learning"
STORE_DIR = ROOT / "stores" / RUN_NAME / "latest"
PLOT_DIR = ROOT / "plots" / RUN_NAME
SITE_DATA = ROOT / "site" / "data" / "coop_grid_q_learning.json"

SIZE = 5
NUM_ENVS = 2048
MAX_STEPS = 48
ITERATIONS = 10
CHUNKS_PER_ITERATION = 6
CHUNK_STEPS = 1000


def state_ids(positions, target, active, size):
    cell_ids = positions[:, :, 1] * size + positions[:, :, 0]
    powers = (size * size) ** np.arange(positions.shape[1], dtype=np.int64)
    position_id = np.sum(cell_ids * powers[None, :], axis=-1)
    target_code = np.where(active, target + 1, 0)
    return target_code * (size * size) ** positions.shape[1] + position_id


def evaluate_policy(q, env):
    size = env.size
    targets = np.asarray(env.targets)
    zones = np.asarray(env.zones)
    moves = np.asarray(MOVES)
    cells = np.array([[x, y] for y in range(size) for x in range(size)], dtype=np.int64)
    cases = []
    case_targets = []
    for target_id in range(len(targets)):
        for first in cells:
            for second in cells:
                cases.append([first, second])
                case_targets.append(target_id)

    positions = np.array(cases, dtype=np.int64)
    target = np.array(case_targets, dtype=np.int64)
    active = np.zeros(len(positions), dtype=bool)
    done = np.zeros(len(positions), dtype=bool)
    collected = np.zeros(len(positions), dtype=bool)
    wrong = np.zeros(len(positions), dtype=bool)
    activated = np.zeros(len(positions), dtype=bool)
    done_step = np.full(len(positions), MAX_STEPS, dtype=np.int64)
    visited = np.zeros(q.shape[1], dtype=bool)
    center = np.array([size // 2, size // 2], dtype=np.int64)

    for step in range(MAX_STEPS):
        mask = ~done
        ids = state_ids(positions, target, active, size)
        visited[ids[mask]] = True
        actions = np.stack([np.argmax(q[agent, ids], axis=-1) for agent in range(q.shape[0])], axis=1)
        next_positions = np.clip(positions + moves[actions], 0, size - 1)
        positions[mask] = next_positions[mask]

        center_hit = np.all(np.all(positions == center[None, None, :], axis=-1), axis=-1)
        activated_now = mask & ~active & center_hit
        active = active | activated_now
        activated = activated | activated_now

        target_cells = targets[target]
        at_target = np.all(positions[:, :, None, :] == target_cells[:, None, :, :], axis=-1)
        all_at_first = np.all(at_target[:, :, 0], axis=-1)
        all_at_second = np.all(at_target[:, :, 1], axis=-1)
        collected_now = mask & active & (all_at_first | all_at_second)

        at_zone = np.any(np.all(positions[:, :, None, :] == zones[None, None, :, :], axis=-1), axis=-1)
        at_selected = np.any(at_target, axis=-1)
        wrong_now = mask & active & np.any(at_zone & ~at_selected, axis=-1)
        timeout_now = mask & (step == MAX_STEPS - 1)
        done_now = collected_now | wrong_now | timeout_now

        collected = collected | collected_now
        wrong = wrong | wrong_now
        done_step[done_now & ~done] = step + 1
        done = done | done_now

    margins = []
    visited_ids = np.flatnonzero(visited)
    for agent in range(q.shape[0]):
        values = np.sort(q[agent, visited_ids], axis=-1)
        margins.append(values[:, -1] - values[:, -2])
    margins = np.concatenate(margins)

    return {
        "cases": int(len(positions)),
        "collection_rate": float(np.mean(collected)),
        "activation_rate": float(np.mean(activated)),
        "wrong_target_rate": float(np.mean(wrong)),
        "timeout_rate": float(np.mean(~collected & ~wrong)),
        "mean_done_steps": float(np.mean(done_step)),
        "mean_collection_steps": float(np.mean(done_step[collected])) if np.any(collected) else float(MAX_STEPS),
        "visited_states": int(np.sum(visited)),
        "mean_q_margin": float(np.mean(margins)) if len(margins) else 0.0,
        "median_q_margin": float(np.median(margins)) if len(margins) else 0.0,
    }


def write_learning_plot(history):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    iterations = np.array([item["iteration"] for item in history])
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(iterations, [item["eval_collection_rate"] for item in history], label="eval collection")
    ax.plot(iterations, [item["eval_activation_rate"] for item in history], label="eval activation")
    ax.plot(iterations, [item["train_collected"] for item in history], label="train collection")
    ax.set_xlabel("iteration")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "learning_rates.png")
    plt.close(fig)


def write_failure_plot(history):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    final = history[-1]
    values = [
        final["eval_collection_rate"],
        final["eval_activation_rate"] - final["eval_collection_rate"],
        final["eval_wrong_target_rate"],
        final["eval_timeout_rate"],
    ]
    labels = ["collected", "activated only", "wrong target", "timeout"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(labels, values, color=["#2f7d4f", "#d8a328", "#9c3d3d", "#6f7785"])
    ax.set_ylabel("fraction of start-target cases")
    ax.set_ylim(0, 1)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "failure_modes.png")
    plt.close(fig)


def write_policy_plot(q, env, active):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    size = env.size
    center = np.array([size // 2, size // 2], dtype=np.int64)
    target = np.zeros(size * size, dtype=np.int64)
    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    agent_cells = np.column_stack([xs.ravel(), ys.ravel()])
    partner = np.repeat(center[None, :], len(agent_cells), axis=0)
    if active:
        partner = np.repeat(np.asarray(env.targets)[0, 0][None, :], len(agent_cells), axis=0)
    positions = np.stack([agent_cells, partner], axis=1)
    ids = state_ids(positions, target, np.full(len(agent_cells), active), size)
    actions = np.argmax(q[0, ids], axis=-1)
    moves = np.asarray(MOVES)[actions]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(color="0.86")
    ax.quiver(agent_cells[:, 0], agent_cells[:, 1], moves[:, 0], moves[:, 1], angles="xy", scale_units="xy", scale=1)
    ax.scatter([center[0]], [center[1]], marker="s", s=120, color="#d8a328")
    if active:
        target_cells = np.asarray(env.targets)[0]
        ax.scatter(target_cells[:, 0], target_cells[:, 1], marker="o", s=120, color="#2f7d4f")
    title = "active_target_policy.png" if active else "inactive_center_policy.png"
    fig.tight_layout()
    fig.savefig(PLOT_DIR / title)
    plt.close(fig)


def write_q_margin_plot(q):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    margins = []
    for agent in range(q.shape[0]):
        values = np.sort(q[agent], axis=-1)
        nonzero = np.any(q[agent] != 0, axis=-1)
        margins.append(values[nonzero, -1] - values[nonzero, -2])
    margins = np.concatenate(margins)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(margins, bins=40, color="#536d8f")
    ax.set_xlabel("top Q minus second Q")
    ax.set_ylabel("states")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "q_margins.png")
    plt.close(fig)


def write_outputs(q, env, history):
    metrics = history[-1].copy()
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(STORE_DIR / "q_values.npz", q=q)
    with open(STORE_DIR / "metrics.json", "w") as f:
        json.dump({"history": history, "final": metrics}, f, indent=2)

    rounded_q = np.round(q, 5).astype(np.float32)
    payload = {
        "name": RUN_NAME,
        "schema_version": 1,
        "env": {
            "size": env.size,
            "num_agents": env.num_agents,
            "num_actions": env.num_actions,
            "num_states": env.num_states,
            "max_steps": env.max_steps,
            "center": [env.size // 2, env.size // 2],
            "moves": np.asarray(MOVES).astype(int).tolist(),
            "targets": np.asarray(env.targets).astype(int).tolist(),
        },
        "policy": {
            "kind": "independent_q_learning",
            "q_shape": list(rounded_q.shape),
            "q": rounded_q.reshape(-1).tolist(),
        },
        "training": {
            "iterations": ITERATIONS,
            "chunks_per_iteration": CHUNKS_PER_ITERATION,
            "chunk_steps": CHUNK_STEPS,
            "num_envs": NUM_ENVS,
        },
        "diagnostics": {
            "history": history,
            "final": metrics,
        },
    }
    with open(SITE_DATA, "w") as f:
        json.dump(payload, f, separators=(",", ":"))


def train():
    env = CooperativeGridWorld(
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
    learner = IndependentQLearning(
        num_agents=env.num_agents,
        num_states=env.num_states,
        num_actions=env.num_actions,
        alpha=0.03,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_steps=ITERATIONS * CHUNKS_PER_ITERATION * CHUNK_STEPS,
    )
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

    q = np.asarray(learner_state.q)
    write_learning_plot(history)
    write_failure_plot(history)
    write_policy_plot(q, env, active=False)
    write_policy_plot(q, env, active=True)
    write_q_margin_plot(q)
    write_outputs(q, env, history)


if __name__ == "__main__":
    train()
