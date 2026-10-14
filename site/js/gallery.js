import {demos} from './demos.js';
import {createPolicy} from './policy.js';
import {createSimulation} from './simulation.js';
import {createRenderer} from './render.js';
import {createInput} from './input.js';

const canvas = document.querySelector('canvas');
const selector = document.querySelector('#demo');
const play = document.querySelector('#play');
const reset = document.querySelector('#reset');
const status = document.querySelector('#status');
const catalogURL = new URL('../assets/manifest.json', import.meta.url);
const catalog = await (await fetch(catalogURL)).json();
const assets = new Map();
const renderer = createRenderer(canvas, assets);
let current;
let input;
let lastTime = null;
let lastMessage = '';
const announce = message => { status.textContent = message; };

async function activate(demo) {
  if (input) { input.clear(); input.selected = null; }
  if (current) current.simulation.suspended = true;
  selector.disabled = play.disabled = reset.disabled = true;
  canvas.style.pointerEvents = 'none';
  canvas.inert = true;
  const manifest = await (await fetch(demo.manifest)).json();
  const bytes = new Uint8Array(await (await fetch(new URL(manifest.policy.file, demo.manifest))).arrayBuffer());
  const environment = demo.environment(manifest.environment.config);
  const view = demo.view(manifest.environment.config, manifest.art);
  const simulation = createSimulation({environment, view, policy: createPolicy(bytes, manifest.policy.shape[1], {
      epsilon: demo.playback.epsilon, numActions: manifest.environment.actions.length,
    }),
    scenarios: manifest.scenarios, sampleScenario: () => environment.sampleScenario(),
    tickSeconds: demo.playback.tickSeconds, terminalSeconds: demo.playback.terminalSeconds,
    stayAction: manifest.environment.stayAction});
  const ids = new Set(simulation.scene().sprites.map(sprite => sprite.asset));
  await Promise.all(catalog.assets.filter(a => ids.has(a.id) && !assets.has(a.id)).map(async asset => {
    const image = new Image();
    image.src = new URL(asset.file, catalogURL).href;
    await image.decode();
    assets.set(asset.id, {...asset, image});
  }));
  simulation.playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  simulation.suspended = document.hidden;
  current = {simulation, view};
  document.querySelector('h1').textContent = demo.name;
  document.querySelector('#task').textContent = demo.description;
  canvas.setAttribute('aria-label', `${demo.name}. Enter selects an agent; arrow keys move it.`);
  play.textContent = simulation.playing ? 'Pause' : 'Play';
  selector.disabled = play.disabled = reset.disabled = false;
  canvas.style.pointerEvents = '';
  canvas.inert = false;
  lastTime = null;
  lastMessage = '';
  announce('');
  renderer.draw(simulation.scene());
}

for (const demo of demos) selector.add(new Option(demo.name, demo.id));
selector.parentElement.hidden = demos.length === 1;
await activate(demos[0]);
input = createInput(canvas, renderer, () => current, announce);
selector.addEventListener('change', () => activate(demos.find(demo => demo.id === selector.value)));
play.addEventListener('click', () => {
  const simulation = current.simulation;
  simulation.playing = !simulation.playing;
  play.textContent = simulation.playing ? 'Pause' : 'Play';
  lastTime = null;
});
reset.addEventListener('click', () => {
  input.clear();
  input.selected = null;
  current.simulation.reset();
  lastTime = null;
  announce('');
});
document.addEventListener('visibilitychange', () => {
  input.clear();
  current.simulation.suspended = document.hidden;
  lastTime = null;
});
new ResizeObserver(() => {
  renderer.resize();
  renderer.draw(current.simulation.scene(), input.overlay());
}).observe(canvas);

function animate(time) {
  const {simulation, view} = current;
  const elapsed = lastTime === null ? 0 : (time - lastTime) / 1000;
  lastTime = time;
  simulation.advance(elapsed);
  const message = view.status(simulation.result);
  if (message !== lastMessage) { announce(message); lastMessage = message; }
  renderer.draw(simulation.scene(), input.overlay());
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
