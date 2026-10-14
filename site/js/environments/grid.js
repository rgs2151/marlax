export function createGrid(config) {
  const {size, center, zones, targets, moves, rewards, timeLimit} = config;
  const equal = (a, b) => a[0] === b[0] && a[1] === b[1];
  return {
    sampleScenario(random = Math.random) {
      return {state: {
        positions: Array.from({length: 2}, () => [Math.floor(random() * size), Math.floor(random() * size)]),
        target: Math.floor(random() * targets.length), active: false, steps: 0,
      }};
    },
    reset: scenario => structuredClone(scenario.state),
    observe(state) {
      const cells = state.positions.map(([x, y]) => y * size + x);
      const code = state.active ? state.target + 1 : 0;
      const index = code * size ** 4 + cells[0] + cells[1] * size ** 2;
      return state.positions.map(() => index);
    },
    step(state, actions) {
      const positions = state.positions.map((p, i) => p.map((v, axis) =>
        Math.max(0, Math.min(size - 1, v + moves[actions[i]][axis]))));
      const activated = !state.active && positions.every(p => equal(p, center));
      const active = state.active || activated;
      const selected = targets[state.target];
      const collected = active && selected.some(t => positions.every(p => equal(p, t)));
      const split = active && !collected && selected.every(t => positions.some(p => equal(p, t)));
      const wrongTarget = active && positions.some(p =>
        zones.some(z => equal(p, z)) && !selected.some(t => equal(p, t)));
      const reward = rewards.step + rewards.activation * activated + rewards.target * collected
        + rewards.split * split + rewards.wrong * wrongTarget;
      const next = {positions, target: state.target, active, steps: state.steps + 1};
      return {
        state: next, reward: positions.map(() => reward),
        terminated: collected || wrongTarget, truncated: next.steps >= timeLimit,
        events: {activated, collected, wrongTarget},
      };
    },
  };
}
