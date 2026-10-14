"""Export the frozen grid policy and small Python transition references."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from marlax.envs.coop_grid import CooperativeGrid, GridState

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=ROOT / 'stores/coop_grid_q_learning/latest/q_values.npz')
    parser.add_argument('--sharing', choices=['independent', 'shared'], default='independent')
    parser.add_argument('--output', type=Path, default=ROOT / 'site/data/grid/v1')
    args = parser.parse_args()
    env = CooperativeGrid()
    params = env.default_params()
    with np.load(args.weights, allow_pickle=False) as arrays:
        q = arrays['q']
    actions = np.argmax(q, axis=-1).astype(np.uint8)
    if args.sharing == 'shared':
        actions = np.repeat(actions, env.num_agents, axis=0)
    starts = [[[1, 0], [3, 4]], [[0, 1], [4, 3]], [[4, 4], [0, 0]],
              [[3, 0], [1, 4]], [[0, 4], [4, 0]], [[0, 2], [4, 2]]]
    scenarios = [{'id': f'target-{i}', 'state': {'positions': positions, 'target': i,
                  'active': False, 'steps': 0}} for i, positions in enumerate(starts)]
    config = {
        'size': env.size, 'center': env.center.tolist(), 'zones': env.zones.tolist(),
        'targets': env.targets.tolist(), 'moves': [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]],
        'timeLimit': params.time_limit,
        'rewards': {'step': params.step_reward, 'activation': params.activation_reward,
                    'target': params.target_reward, 'wrong': params.wrong_target_reward,
                    'split': params.split_target_reward},
    }
    manifest = {
        'format': 'marlax.browser', 'version': 1, 'id': 'grid', 'revision': 'v1',
        'environment': {'id': 'coop_grid', 'version': 1, 'config': config,
                        'coordinates': 'x_right_y_up', 'observationSchema': env.spec.observation_schema,
                        'actions': ['stay', 'south', 'north', 'west', 'east'], 'stayAction': 0},
        'policy': {'file': 'actions.u8', 'dtype': 'uint8', 'shape': list(actions.shape),
                   'order': 'C', 'tieBreak': 'lowest_action_index'},
        'scenarios': scenarios, 'tickSeconds': 0.18,
        'art': {'board': 'circuit', 'meeting': 'portal', 'target': 'beacon', 'agents': ['orbit', 'dart']},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'actions.u8').write_bytes(actions.tobytes())
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    cases = [
        ('clip', [[0, 0], [4, 4]], False, 0, [3, 4]),
        ('activate', [[2, 2], [2, 1]], False, 0, [0, 2]),
        ('collect', [[2, 3], [2, 4]], True, 4, [2, 0]),
        ('split', [[2, 4], [4, 2]], True, 0, [0, 0]),
        ('wrong', [[2, 1], [4, 2]], True, 0, [1, 0]),
        ('timeout', [[0, 0], [1, 1]], False, 47, [0, 0]),
        ('collect-at-limit', [[2, 4], [2, 4]], True, 47, [0, 0]),
    ]
    fixtures = []
    for name, positions, active, steps, joint in cases:
        state = GridState(jnp.array(positions), jnp.array(0), jnp.array(active), jnp.array(steps))
        after, result = env.step(jax.random.PRNGKey(0), state, jnp.array(joint), params)
        fixtures.append({
            'name': name, 'state': {'positions': positions, 'target': 0, 'active': active, 'steps': steps},
            'actions': joint, 'observations': env.observe(state, params).obs.tolist(),
            'result': {'state': {'positions': after.positions.tolist(), 'target': int(after.target),
                                'active': bool(after.active), 'steps': int(after.steps)},
                       'reward': result.reward.tolist(), 'terminated': bool(result.terminated),
                       'truncated': bool(result.truncated),
                       'events': {'activated': bool(result.metrics['activated']),
                                  'collected': bool(result.metrics['collected']),
                                  'wrongTarget': bool(result.metrics['wrong_target'])}},
        })
    (ROOT / 'tests/fixtures/grid.json').write_text(json.dumps(fixtures, indent=2) + '\n')


if __name__ == '__main__':
    main()
