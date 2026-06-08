import json

import numpy as np

from config import (
    CHUNK_STEPS,
    CHUNKS_PER_ITERATION,
    ITERATIONS,
    NUM_ENVS,
    SCENARIOS,
    SITE_DATA,
    STORE_DIR,
    run_metadata,
)
from marlax.envs.coop_grid import MOVES


def write_outputs(q, env, history, final_metrics, scenario_results):
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(STORE_DIR / "q_values.npz", q=q)
    with open(STORE_DIR / "metrics.json", "w") as f:
        json.dump({
            "history": history,
            "final": final_metrics,
            "showcase": scenario_results,
        }, f, indent=2)
    with open(STORE_DIR / "metadata.json", "w") as f:
        json.dump(run_metadata(), f, indent=2)

    rounded_q = np.round(q, 5).astype(np.float32)
    payload = {
        "name": "coop_grid_q_learning",
        "schema_version": 2,
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
            "mode": "greedy",
            "q_shape": list(rounded_q.shape),
            "q": rounded_q.reshape(-1).tolist(),
        },
        "showcase": {
            "mode": "greedy_cycle",
            "scenarios": SCENARIOS,
            "scenario_results": [
                {
                    "name": item["name"],
                    "target_id": item["target_id"],
                    "starts": item["starts"],
                    "status": item["status"],
                    "steps": item["steps"],
                }
                for item in scenario_results
            ],
        },
        "training": {
            "iterations": ITERATIONS,
            "chunks_per_iteration": CHUNKS_PER_ITERATION,
            "chunk_steps": CHUNK_STEPS,
            "num_envs": NUM_ENVS,
        },
        "diagnostics": {
            "history": history,
            "final": final_metrics,
        },
    }
    with open(SITE_DATA, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
