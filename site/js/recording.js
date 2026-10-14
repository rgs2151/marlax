export function recordTrajectory(environment, policy, scenario, {tickSeconds, maxSteps}) {
  let state = environment.reset(scenario);
  const frames = [{state}];
  for (let step = 0; step < maxSteps; step += 1) {
    const actions = policy.act(environment.observe(state));
    const result = environment.step(state, actions);
    frames.push({...result, actions});
    state = result.state;
    if (result.terminated || result.truncated) break;
  }
  return {format: 'marlax.trajectory', version: 1, tickSeconds, frames};
}

export function createReplay(recording, view, terminalSeconds = 0.6) {
  const {frames, tickSeconds} = recording;
  const headings = [[]];
  for (let i = 1; i < frames.length; i += 1) {
    const scene = view.scene(frames[i - 1].state, frames[i].state, 1, headings[i - 1]);
    headings.push(scene.sprites.filter(p => p.agent !== undefined).map(p => p.heading));
  }
  const movementDuration = (frames.length - 1) * tickSeconds;
  return {
    duration: movementDuration + terminalSeconds,
    keyTimes: frames.map((_, index) => index * tickSeconds),
    scene(seconds) {
      if (seconds >= movementDuration) {
        const last = frames.length - 1;
        return view.scene(frames[last].state, frames[last].state, 1, headings[last]);
      }
      const index = Math.floor(seconds / tickSeconds);
      const fraction = seconds / tickSeconds - index;
      return view.scene(frames[index].state, frames[index + 1].state, fraction, headings[index]);
    },
  };
}
