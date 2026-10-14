import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createGrid} from '../../site/js/environments/grid.js';
import {createGridView} from '../../site/js/views/grid.js';
import {createPolicy} from '../../site/js/policy.js';
import {createSimulation} from '../../site/js/simulation.js';
import {recordTrajectory, createReplay} from '../../site/js/recording.js';
import {exportAnimatedSVG} from '../../site/js/svg.js';
import {createProjection} from '../../site/js/render.js';

const url = new URL('../../site/data/grid/v1/manifest.json', import.meta.url);
const manifest = JSON.parse(await readFile(url));
const bytes = new Uint8Array(await readFile(new URL(manifest.policy.file, url)));
const environment = createGrid(manifest.environment.config);
const policy = createPolicy(bytes, manifest.policy.shape[1]);
const view = createGridView(manifest.environment.config, manifest.art);
const session = () => createSimulation({environment, policy, view, scenarios: manifest.scenarios,
  tickSeconds: manifest.tickSeconds, stayAction: 0});

test('grid transition agrees with Python fixtures including terminal rewards', async () => {
  const fixtures = JSON.parse(await readFile(new URL('../fixtures/grid.json', import.meta.url)));
  for (const fixture of fixtures) {
    const before = structuredClone(fixture.state);
    assert.deepEqual(environment.observe(before), fixture.observations, fixture.name);
    const result = environment.step(before, fixture.actions);
    result.reward.forEach((value, i) => assert.ok(Math.abs(value - fixture.result.reward[i]) < 1e-6));
    assert.deepEqual({...result, reward: []}, {...fixture.result, reward: []}, fixture.name);
    assert.deepEqual(before, fixture.state);
  }
});

test('clock and episode cycling do not depend on display rate', () => {
  const results = [30, 60, 120, 144].map(fps => {
    const sim = session();
    for (let i = 0; i < fps * 60; i += 1) sim.advance(1 / fps);
    return {state: sim.state, scenario: sim.scenarioIndex, ended: sim.ended};
  });
  for (const result of results) assert.deepEqual(result, results[0]);
  const sim = session();
  sim.suspended = true;
  sim.advance(60);
  assert.equal(sim.state.steps, 0);
  sim.suspended = false;
  sim.advance(10);
  assert.ok(sim.state.steps <= 4);
  sim.advance(10);
  assert.equal(sim.scenarioIndex, 1);
  assert.ok(sim.state.steps <= 3);
});

test('placement holds one tick while partner acts, including paused drop and terminal hold', () => {
  const sim = session();
  sim.hold(0);
  sim.place(0, [2.1, 2.2]);
  sim.release();
  sim.advance(.18);
  assert.deepEqual(sim.state.positions[0], [2, 2]);
  assert.notDeepEqual(sim.state.positions[1], manifest.scenarios[0].state.positions[1]);
  assert.equal(sim.held, null);
  sim.playing = false;
  sim.hold(0);
  sim.place(0, [1.2, .2]);
  sim.release();
  const steps = sim.state.steps;
  sim.advance(10);
  assert.equal(sim.state.steps, steps);
  assert.deepEqual(sim.state.positions[0], [1, 0]);
  sim.playing = true;
  sim.advance(.18);
  assert.deepEqual(sim.state.positions[0], [1, 0]);
  sim.state = {...sim.state, active: true};
  sim.hold(0);
  sim.place(0, [2, 0]);
  sim.advance(.18);
  assert.equal(sim.result.events.wrongTarget, true);
  const end = structuredClone(sim.state);
  sim.place(0, [0, 0]);
  sim.advance(10);
  assert.deepEqual(sim.state, end);
  sim.release();
  sim.advance(.6);
  assert.equal(sim.state.steps, 0);
});

test('recordings finish all curated starts and render without a policy', async () => {
  const expected = [5, 5, 6, 5, 6, 4];
  for (const [index, scenario] of manifest.scenarios.entries()) {
    const recording = recordTrajectory(environment, policy, scenario, {tickSeconds: .18, maxSteps: 48});
    assert.equal(recording.frames.length - 1, expected[index]);
    assert.equal(recording.frames.at(-1).events.collected, true);
    const replay = createReplay(JSON.parse(JSON.stringify(recording)), view);
    assert.deepEqual(replay.scene(replay.duration).sprites.filter(p => p.agent !== undefined).map(p => p.position), recording.frames.at(-1).state.positions);
  }
  const recording = recordTrajectory(environment, policy, manifest.scenarios[0], {tickSeconds: .18, maxSteps: 48});
  const catalogURL = new URL('../../site/assets/manifest.json', import.meta.url);
  const catalog = JSON.parse(await readFile(catalogURL));
  const assets = new Map(await Promise.all(catalog.assets.map(async asset =>
    [asset.id, {...asset, svg: await readFile(new URL(asset.file, catalogURL), 'utf8')}])));
  const svg = exportAnimatedSVG(createReplay(recording, view), assets);
  assert.ok(svg.includes('<animateTransform'));
  assert.ok(svg.includes('id="asset-circuit"'));
  assert.ok(!svg.includes('<script') && !svg.includes('data:') && !svg.includes('NaN'));
  assert.ok([...svg.matchAll(/href="([^"]+)"/g)].every(match => match[1].startsWith('#')));
  const projection = createProjection(view.scene(recording.frames[0].state, recording.frames[0].state, 1).bounds, 390, 400);
  const point = [1.25, 3.75];
  const roundTrip = projection.toWorld(projection.toScreen(point));
  point.forEach((v, i) => assert.ok(Math.abs(v - roundTrip[i]) < 1e-10));
});
