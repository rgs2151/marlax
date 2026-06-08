from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "coop_grid_q_learning"
STORE_DIR = ROOT / "stores" / RUN_NAME / "latest"
PLOT_DIR = ROOT / "plots" / RUN_NAME / "latest"
SITE_DATA = ROOT / "site" / "data" / "coop_grid_q_learning.json"

SIZE = 5
NUM_ENVS = 2048
MAX_STEPS = 48
ITERATIONS = 10
CHUNKS_PER_ITERATION = 6
CHUNK_STEPS = 1000

SCENARIOS = [
    {"name": "target_0_cross_edges", "target_id": 0, "starts": [[1, 0], [3, 4]]},
    {"name": "target_1_right_bottom", "target_id": 1, "starts": [[0, 1], [4, 3]]},
    {"name": "target_2_bottom_left", "target_id": 2, "starts": [[4, 4], [0, 0]]},
    {"name": "target_3_top_left", "target_id": 3, "starts": [[3, 0], [1, 4]]},
    {"name": "target_4_vertical", "target_id": 4, "starts": [[0, 4], [4, 0]]},
    {"name": "target_5_horizontal", "target_id": 5, "starts": [[0, 2], [4, 2]]},
]


def run_metadata():
    return {
        "run_name": RUN_NAME,
        "size": SIZE,
        "num_envs": NUM_ENVS,
        "max_steps": MAX_STEPS,
        "iterations": ITERATIONS,
        "chunks_per_iteration": CHUNKS_PER_ITERATION,
        "chunk_steps": CHUNK_STEPS,
        "scenarios": SCENARIOS,
    }
