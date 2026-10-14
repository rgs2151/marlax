export function createSimulation({environment, policy, view, scenarios, tickSeconds, stayAction, terminalSeconds = 0.6}) {
  const session = {
    state: null, previous: null, result: null, headings: [],
    scenarioIndex: 0, accumulator: 0, terminalElapsed: 0,
    playing: true, suspended: false, held: null, pending: null, releasing: false,
    reset(index = this.scenarioIndex) {
      this.scenarioIndex = index;
      this.state = environment.reset(scenarios[index]);
      this.previous = this.state;
      this.result = null;
      this.headings = [];
      this.accumulator = this.terminalElapsed = 0;
      this.held = this.pending = null;
      this.releasing = false;
    },
    get ended() { return Boolean(this.result?.terminated || this.result?.truncated); },
    hold(agent) { if (!this.ended) { this.held = agent; this.releasing = false; } },
    place(agent, point) {
      if (this.ended) return;
      this.pending = {agent, point};
      if (!this.playing) {
        this.state = view.place(this.state, agent, point);
        this.previous = this.state;
      }
    },
    release() {
      if (this.ended) { this.held = null; this.pending = null; return; }
      this.releasing = this.held !== null;
    },
    cancel() {
      this.pending = null;
      this.held = null;
      this.releasing = false;
    },
    tick() {
      if (this.ended) return;
      let placed = this.state;
      if (this.pending) placed = view.place(placed, this.pending.agent, this.pending.point);
      const actions = policy.act(environment.observe(placed));
      if (this.held !== null) actions[this.held] = stayAction;
      this.previous = placed;
      this.result = environment.step(placed, actions);
      this.state = this.result.state;
      const scene = view.scene(this.previous, this.state, 1, this.headings);
      this.headings = scene.sprites.filter(p => p.agent !== undefined).map(p => p.heading);
      if (this.releasing) this.cancel();
    },
    advance(seconds, tickBudget = 4) {
      if (!this.playing || this.suspended) return;
      if (this.ended) {
        if (this.held !== null) return;
        this.terminalElapsed += seconds;
        if (this.terminalElapsed + 1e-10 >= terminalSeconds) {
          const remaining = Math.max(0, this.terminalElapsed - terminalSeconds);
          this.reset((this.scenarioIndex + 1) % scenarios.length);
          this.advance(remaining, tickBudget);
        }
        return;
      }
      this.accumulator += seconds;
      let ticks = 0;
      while (this.accumulator + 1e-10 >= tickSeconds && ticks < tickBudget && !this.ended) {
        this.accumulator = Math.max(0, this.accumulator - tickSeconds);
        this.tick();
        ticks += 1;
      }
      if (this.ended) {
        const remaining = this.accumulator;
        this.accumulator = 0;
        this.advance(remaining, tickBudget - ticks);
      } else if (ticks === tickBudget && this.accumulator >= tickSeconds) this.accumulator %= tickSeconds;
    },
    scene() {
      const fraction = this.ended || !this.playing ? 1 : this.accumulator / tickSeconds;
      return view.scene(this.previous, this.state, fraction, this.headings);
    },
  };
  session.reset(0);
  return session;
}
