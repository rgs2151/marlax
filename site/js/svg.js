import {createProjection} from './render.js';

const number = value => Number(value.toFixed(5));
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;');

export function exportAnimatedSVG(replay, assets, {width = 800, height = 800, fps = 30, title = 'MARLAX simulation'} = {}) {
  const count = Math.ceil(replay.duration * fps);
  const times = [...new Set([
    ...Array.from({length: count + 1}, (_, i) => i * replay.duration / count),
    ...replay.keyTimes,
  ])].sort((a, b) => a - b);
  const scenes = times.map(time => replay.scene(time));
  const projection = createProjection(scenes[0].bounds, width, height);
  const keys = times.map(time => number(time / replay.duration)).join(';');
  const duration = `${number(replay.duration)}s`;
  const animation = (attribute, values, mode = 'linear') =>
    values.every(value => value === values[0]) ? '' : `<animate attributeName="${attribute}" values="${values.join(';')}" keyTimes="${keys}" dur="${duration}" repeatCount="indefinite" calcMode="${mode}"/>`;
  const transform = (type, values, mode = 'linear') =>
    values.every(value => value === values[0]) ? '' : `<animateTransform attributeName="transform" type="${type}" values="${values.join(';')}" keyTimes="${keys}" dur="${duration}" repeatCount="indefinite" calcMode="${mode}"/>`;
  const used = [...new Set(scenes[0].sprites.map(s => s.asset))];
  const defs = used.map(id => {
    const asset = assets.get(id);
    const body = asset.svg.slice(asset.svg.indexOf('>') + 1, asset.svg.lastIndexOf('</svg>'))
      .replace(/<(title|desc)\b[^>]*>[\s\S]*?<\/\1>/g, '')
      .replace(/var\(--\w+,\s*(#[\da-fA-F]+)\)/g, '$1');
    return `<g id="asset-${id}">${body}</g>`;
  }).join('');
  const pieces = scenes[0].sprites.map((sprite, index) => {
    const samples = scenes.map(scene => scene.sprites[index]);
    const asset = assets.get(sprite.asset);
    const positions = samples.map(p => projection.toScreen(p.position).map(number).join(' '));
    const angles = samples.map(p => number(p.heading * 180 / Math.PI));
    const opacity = samples.map(p => p.opacity);
    const sx = sprite.width * projection.scale / asset.viewBox[2];
    const sy = sprite.height * projection.scale / asset.viewBox[3];
    return `<g transform="translate(${positions[0]})">${transform('translate', positions)}<g transform="rotate(${angles[0]})">${transform('rotate', angles, 'discrete')}<g opacity="${opacity[0]}">${animation('opacity', opacity, 'discrete')}<g transform="scale(${sx} ${sy}) translate(${-asset.anchor[0]} ${-asset.anchor[1]})"><use href="#asset-${sprite.asset}"/></g></g></g></g>`;
  }).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="clip-title"><title id="clip-title">${escape(title)}</title><rect width="100%" height="100%" fill="#ffffff"/><defs>${defs}</defs>${pieces}</svg>\n`;
}
