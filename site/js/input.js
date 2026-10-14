export function createInput(canvas, renderer, current, announce) {
  const input = {
    pointerId: null, selected: null, preview: null, offset: [0, 0],
    lastCandidates: '', cycle: -1,
    clear() {
      const pointer = this.pointerId;
      this.pointerId = null;
      this.preview = null;
      canvas.style.cursor = '';
      current().simulation.cancel();
      if (pointer !== null && canvas.hasPointerCapture(pointer)) canvas.releasePointerCapture(pointer);
    },
    overlay() {
      const {simulation, view} = current();
      if (simulation.ended) return {selected: this.selected};
      let landing = null;
      if (this.preview) {
        const placed = view.place(simulation.state, this.selected, this.preview);
        landing = view.scene(placed, placed, 1).sprites.find(p => p.agent === this.selected).position;
      }
      return {selected: this.selected, preview: this.preview, landing};
    },
  };
  const request = event => {
    const {simulation} = current();
    if (simulation.ended) return;
    const world = renderer.projection.toWorld(renderer.pointer(event));
    input.preview = world.map((v, axis) => v + input.offset[axis]);
    simulation.place(input.selected, input.preview);
  };
  canvas.addEventListener('pointerdown', event => {
    const {simulation} = current();
    if (event.button !== 0 || input.pointerId !== null || simulation.ended) return;
    const point = renderer.pointer(event);
    const candidates = renderer.hits(point);
    if (!candidates.length) return;
    event.preventDefault();
    const key = candidates.map(p => p.agent).join(',');
    input.cycle = key === input.lastCandidates ? (input.cycle + 1) % candidates.length : 0;
    input.lastCandidates = key;
    const sprite = candidates[input.cycle];
    input.selected = sprite.agent;
    input.pointerId = event.pointerId;
    const world = renderer.projection.toWorld(point);
    input.offset = sprite.position.map((v, axis) => v - world[axis]);
    simulation.hold(input.selected);
    canvas.style.cursor = 'grabbing';
    canvas.setPointerCapture(event.pointerId);
    canvas.focus({preventScroll: true});
    announce(`Agent ${input.selected + 1} selected.`);
    request(event);
  });
  canvas.addEventListener('pointermove', event => {
    if (event.pointerId === input.pointerId) request(event);
  });
  canvas.addEventListener('pointerup', event => {
    if (event.pointerId !== input.pointerId) return;
    request(event);
    current().simulation.release();
    input.pointerId = null;
    input.preview = null;
    canvas.style.cursor = '';
    canvas.releasePointerCapture(event.pointerId);
  });
  for (const name of ['pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(name, event => {
      if (event.pointerId === input.pointerId) input.clear();
    });
  }
  canvas.addEventListener('keydown', event => {
    const {simulation, view} = current();
    if (event.key === 'Escape') { input.clear(); input.selected = null; return; }
    if (input.pointerId !== null || simulation.ended) return;
    const pieces = renderer.scene.sprites.filter(p => p.agent !== undefined);
    if (event.key === 'Enter') {
      input.selected = ((input.selected ?? -1) + 1) % pieces.length;
      announce(`Agent ${input.selected + 1} selected.`);
      event.preventDefault();
    }
    const directions = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1]};
    if (directions[event.key] && input.selected !== null) {
      event.preventDefault();
      simulation.hold(input.selected);
      const state = simulation.pending ? view.place(simulation.state, simulation.pending.agent, simulation.pending.point) : simulation.state;
      simulation.place(input.selected, view.nudge(state, input.selected, directions[event.key]));
      simulation.release();
    }
  });
  return input;
}
