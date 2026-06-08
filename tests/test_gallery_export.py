import json
from pathlib import Path


def load_export():
    path = Path("site/data/coop_grid_q_learning.json")
    return json.loads(path.read_text())


def test_exported_q_shape_matches_payload():
    data = load_export()
    shape = data["policy"]["q_shape"]
    expected = 1
    for value in shape:
        expected *= value

    assert len(data["policy"]["q"]) == expected


def test_showcase_scenarios_are_valid():
    data = load_export()
    env = data["env"]
    scenarios = data["showcase"]["scenarios"]

    assert data["showcase"]["mode"] == "greedy_cycle"
    assert data["policy"]["mode"] == "greedy"
    assert len(scenarios) > 0

    for scenario in scenarios:
        assert 0 <= scenario["target_id"] < len(env["targets"])
        assert len(scenario["starts"]) == env["num_agents"]
        for start in scenario["starts"]:
            assert len(start) == 2
            assert 0 <= start[0] < env["size"]
            assert 0 <= start[1] < env["size"]
