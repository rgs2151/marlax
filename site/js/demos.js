import {createGrid} from './environments/grid.js';
import {createGridView} from './views/grid.js';

export const demos = [{
  id: 'grid', name: 'Cooperative grid',
  description: 'Meet in the centre, then reach the same red target together.',
  manifest: new URL('../data/grid/v1/manifest.json', import.meta.url),
  environment: createGrid, view: createGridView,
}];
