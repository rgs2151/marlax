import {readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
import {demos} from '../site/js/demos.js';
import {createPolicy} from '../site/js/policy.js';
import {recordTrajectory, createReplay} from '../site/js/recording.js';
import {exportAnimatedSVG} from '../site/js/svg.js';

const [id = 'grid', output = '/tmp/marlax-grid.svg', scenario = '0', trajectoryFile] = process.argv.slice(2);
const demo = demos.find(item => item.id === id);
let recording;
if (trajectoryFile) {
  recording = JSON.parse(await readFile(trajectoryFile, 'utf8'));
} else {
  const manifest = JSON.parse(await readFile(demo.manifest, 'utf8'));
  const bytes = new Uint8Array(await readFile(new URL(manifest.policy.file, demo.manifest)));
  recording = recordTrajectory(demo.environment(manifest.environment.config), createPolicy(bytes, manifest.policy.shape[1]),
    manifest.scenarios[Number(scenario)], {tickSeconds: manifest.tickSeconds, maxSteps: manifest.environment.config.timeLimit});
  recording.demo = demo.id;
  recording.environment = manifest.environment;
  recording.art = manifest.art;
}
const view = demo.view(recording.environment.config, recording.art);
const catalogURL = new URL('../site/assets/manifest.json', import.meta.url);
const catalog = JSON.parse(await readFile(catalogURL, 'utf8'));
const replay = createReplay(recording, view);
const ids = new Set(replay.scene(0).sprites.map(sprite => sprite.asset));
const assets = new Map(await Promise.all(catalog.assets.filter(a => ids.has(a.id)).map(async asset =>
  [asset.id, {...asset, svg: await readFile(new URL(asset.file, catalogURL), 'utf8')}])));
const destination = pathToFileURL(resolve(output));
await writeFile(destination, exportAnimatedSVG(replay, assets, {title: demo.name}));
await writeFile(new URL(`${destination.href}.json`), JSON.stringify(recording) + '\n');
