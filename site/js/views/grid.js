export function createGridView(config, art) {
  const {size, center, targets} = config;
  const bounds = [-1.35, -1.35, size + 0.35, size + 0.35];
  const piece = (id, asset, position, width, heading = 0) =>
    ({id, asset, position, width, height: width, heading, opacity: 1});
  return {
    place(state, agent, point) {
      const positions = state.positions.map(p => p.slice());
      positions[agent] = point.map(v => Math.round(Math.max(0, Math.min(size - 1, v))));
      return {...state, positions};
    },
    nudge(state, agent, direction) {
      return state.positions[agent].map((v, axis) => v + direction[axis]);
    },
    scene(previous, state, fraction, headings = []) {
      const sprites = [piece('board', art.board, center, size * 512 / 400),
        piece('meeting', art.meeting, center, 1.35)];
      targets[state.target].forEach((target, i) => sprites.push({
        ...piece(`target-${i}`, art.target, target, 1.35), opacity: (fraction < 1 ? previous.active : state.active) ? 1 : 0,
      }));
      state.positions.forEach((point, i) => {
        const start = previous.positions[i];
        const dx = point[0] - start[0], dy = point[1] - start[1];
        const heading = dx || dy ? Math.atan2(dx, dy) : (headings[i] ?? 0);
        sprites.push({...piece(`agent-${i}`, art.agents[i],
          point.map((v, axis) => start[axis] + (v - start[axis]) * fraction), 1, heading), agent: i});
      });
      return {bounds, sprites};
    },
    status(result) {
      if (result?.events.collected) return 'Target reached.';
      if (result?.events.wrongTarget) return 'Wrong target. Try again.';
      if (result?.truncated) return 'Time is up.';
      return '';
    },
  };
}
