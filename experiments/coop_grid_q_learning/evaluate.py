import numpy as np
from tqdm import tqdm

from marlax.envs.coop_grid import MOVES


def state_ids(positions, target, active, size):
    cell_ids = positions[:, :, 1] * size + positions[:, :, 0]
    powers = (size * size) ** np.arange(positions.shape[1], dtype=np.int64)
    position_id = np.sum(cell_ids * powers[None, :], axis=-1)
    target_code = np.where(active, target + 1, 0)
    return target_code * (size * size) ** positions.shape[1] + position_id


def greedy_actions(q, state_id):
    return np.stack([
        np.argmax(q[agent, state_id], axis=-1)
        for agent in range(q.shape[0])
    ], axis=-1)


def evaluate_policy(q, env, show_progress=False):
    targets = np.asarray(env.targets)
    zones = np.asarray(env.zones)
    moves = np.asarray(MOVES)
    cells = np.array([[x, y] for y in range(env.size) for x in range(env.size)], dtype=np.int64)
    iterator = range(len(targets))
    if show_progress:
        iterator = tqdm(iterator, desc="Evaluating greedy policy", unit="target")

    all_collected = []
    all_activated = []
    all_wrong = []
    all_done_steps = []
    visited = np.zeros(q.shape[1], dtype=bool)

    for target_id in iterator:
        positions = np.array([
            [first, second]
            for first in cells
            for second in cells
        ], dtype=np.int64)
        target = np.full(len(positions), target_id, dtype=np.int64)
        active = np.zeros(len(positions), dtype=bool)
        done = np.zeros(len(positions), dtype=bool)
        collected = np.zeros(len(positions), dtype=bool)
        wrong = np.zeros(len(positions), dtype=bool)
        activated = np.zeros(len(positions), dtype=bool)
        done_step = np.full(len(positions), env.max_steps, dtype=np.int64)
        center = np.array([env.size // 2, env.size // 2], dtype=np.int64)

        for step in range(env.max_steps):
            mask = ~done
            ids = state_ids(positions, target, active, env.size)
            visited[ids[mask]] = True
            actions = greedy_actions(q, ids)
            next_positions = np.clip(positions + moves[actions], 0, env.size - 1)
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
            timeout_now = mask & (step == env.max_steps - 1)
            done_now = collected_now | wrong_now | timeout_now

            collected = collected | collected_now
            wrong = wrong | wrong_now
            done_step[done_now & ~done] = step + 1
            done = done | done_now

        all_collected.append(collected)
        all_activated.append(activated)
        all_wrong.append(wrong)
        all_done_steps.append(done_step)

    collected = np.concatenate(all_collected)
    activated = np.concatenate(all_activated)
    wrong = np.concatenate(all_wrong)
    done_step = np.concatenate(all_done_steps)
    margins = []
    visited_ids = np.flatnonzero(visited)
    for agent in range(q.shape[0]):
        values = np.sort(q[agent, visited_ids], axis=-1)
        margins.append(values[:, -1] - values[:, -2])
    margins = np.concatenate(margins)

    return {
        "cases": int(len(collected)),
        "collection_rate": float(np.mean(collected)),
        "activation_rate": float(np.mean(activated)),
        "wrong_target_rate": float(np.mean(wrong)),
        "timeout_rate": float(np.mean(~collected & ~wrong)),
        "mean_done_steps": float(np.mean(done_step)),
        "mean_collection_steps": float(np.mean(done_step[collected])) if np.any(collected) else float(env.max_steps),
        "visited_states": int(np.sum(visited)),
        "mean_q_margin": float(np.mean(margins)) if len(margins) else 0.0,
        "median_q_margin": float(np.median(margins)) if len(margins) else 0.0,
    }


def rollout_scenario(q, env, scenario):
    targets = np.asarray(env.targets)
    zones = np.asarray(env.zones)
    moves = np.asarray(MOVES)
    positions = np.array([scenario["starts"]], dtype=np.int64)
    target = np.array([scenario["target_id"]], dtype=np.int64)
    active = np.array([False])
    center = np.array([env.size // 2, env.size // 2], dtype=np.int64)
    path = [positions[0].copy()]
    status = "timeout"

    for step in range(env.max_steps):
        ids = state_ids(positions, target, active, env.size)
        actions = greedy_actions(q, ids)
        positions = np.clip(positions + moves[actions], 0, env.size - 1)
        path.append(positions[0].copy())

        center_hit = np.all(np.all(positions == center[None, None, :], axis=-1), axis=-1)
        active = active | (~active & center_hit)

        target_cells = targets[target]
        at_target = np.all(positions[:, :, None, :] == target_cells[:, None, :, :], axis=-1)
        all_at_first = np.all(at_target[:, :, 0], axis=-1)
        all_at_second = np.all(at_target[:, :, 1], axis=-1)
        if bool(active[0] and (all_at_first[0] or all_at_second[0])):
            status = "collected"
            break

        at_zone = np.any(np.all(positions[:, :, None, :] == zones[None, None, :, :], axis=-1), axis=-1)
        at_selected = np.any(at_target, axis=-1)
        wrong = active & np.any(at_zone & ~at_selected, axis=-1)
        if bool(wrong[0]):
            status = "wrong_target"
            break

    return {
        "name": scenario["name"],
        "target_id": int(scenario["target_id"]),
        "starts": scenario["starts"],
        "status": status,
        "steps": len(path) - 1,
        "path": np.asarray(path, dtype=np.int64).tolist(),
    }


def rollout_scenarios(q, env, scenarios, show_progress=False):
    iterator = scenarios
    if show_progress:
        iterator = tqdm(iterator, desc="Rolling showcase scenarios", unit="scenario")
    return [rollout_scenario(q, env, scenario) for scenario in iterator]
