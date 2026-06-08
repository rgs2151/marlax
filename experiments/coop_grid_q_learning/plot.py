import matplotlib.pyplot as plt
import numpy as np

from config import PLOT_DIR
from evaluate import state_ids
from marlax.envs.coop_grid import MOVES


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


def write_failure_plot(final_metrics):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    values = [
        final_metrics["collection_rate"],
        final_metrics["activation_rate"] - final_metrics["collection_rate"],
        final_metrics["wrong_target_rate"],
        final_metrics["timeout_rate"],
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
    center = np.array([env.size // 2, env.size // 2], dtype=np.int64)
    target = np.zeros(env.size * env.size, dtype=np.int64)
    xs, ys = np.meshgrid(np.arange(env.size), np.arange(env.size))
    agent_cells = np.column_stack([xs.ravel(), ys.ravel()])
    partner = np.repeat(center[None, :], len(agent_cells), axis=0)
    if active:
        partner = np.repeat(np.asarray(env.targets)[0, 0][None, :], len(agent_cells), axis=0)
    positions = np.stack([agent_cells, partner], axis=1)
    ids = state_ids(positions, target, np.full(len(agent_cells), active), env.size)
    actions = np.argmax(q[0, ids], axis=-1)
    moves = np.asarray(MOVES)[actions]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
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


def write_showcase_rollout_plot(env, scenario_results):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    cols = 3
    rows = int(np.ceil(len(scenario_results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    center = np.array([env.size // 2, env.size // 2])

    for ax, result in zip(axes, scenario_results):
        path = np.asarray(result["path"])
        target_cells = np.asarray(env.targets)[result["target_id"]]
        ax.set_xlim(-0.5, env.size - 0.5)
        ax.set_ylim(-0.5, env.size - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.grid(color="0.88")
        ax.plot(path[:, 0, 0], path[:, 0, 1], color="#8f2434", linewidth=2)
        ax.plot(path[:, 1, 0], path[:, 1, 1], color="#233f83", linewidth=2)
        ax.scatter(path[0, :, 0], path[0, :, 1], marker="x", color="black")
        ax.scatter([center[0]], [center[1]], marker="s", s=90, color="#d8a328")
        ax.scatter(target_cells[:, 0], target_cells[:, 1], marker="o", s=90, color="#2f7d4f")
        ax.set_title(f"{result['name']} ({result['status']}, {result['steps']} steps)", fontsize=8)

    for ax in axes[len(scenario_results):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "showcase_rollouts.png")
    plt.close(fig)
