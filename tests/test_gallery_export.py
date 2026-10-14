import json
from pathlib import Path

import numpy as np


def test_browser_actions_match_saved_policy():
    path = Path('site/data/grid/v1')
    data = json.loads((path / 'manifest.json').read_text())
    actions = np.fromfile(path / data['policy']['file'], dtype=np.uint8).reshape(data['policy']['shape'])
    with np.load('stores/coop_grid_q_learning/latest/q_values.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(actions, np.argmax(saved['q'], axis=-1))
    assert all(len(s['state']['positions']) == len(actions) for s in data['scenarios'])
