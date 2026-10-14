# JavaScript runtime

The gallery uses native ES modules and Canvas 2D. Its first demo is the existing
cooperative grid, running the saved policy with Python-equivalent world
transitions. Live playback samples independent starting cells and a target pair
at each reset. Each agent chooses a uniformly random action with probability 0.18
and otherwise follows the frozen greedy policy. Playback uses 650 ms per move
and a 1.5-second terminal hold. These live settings are in `site/js/demos.js`;
recorded exports retain explicit starts and deterministic greedy execution. Artwork follows [the approved visual standard](aesthetics.md).
This guide describes the implemented interfaces; [the audit](javascript-redesign.md)
records the rationale. Docking and orbit were rejected proposals and are not implemented.
See [Demo candidates](demo-candidates.md) for the current research shortlist.

## Run locally

From the repository root:

```sh
python -m http.server 8876 --directory site
```

Open `http://localhost:8876/`. For navigation back to Sphinx documentation, serve the
assembled Pages directory instead: documentation lives at its root and the gallery
at `/gallery/`. All runtime resources resolve relative to their module or manifest,
including when deployed below `/marlax/`.

There is no npm install, bundle step or runtime dependency. `site/package.json`
marks JavaScript files as modules and provides a local test command. Pages still
copies the static site; its deployment workflow does not run tests.

## Add a demo

1. Implement a DOM-free environment in `site/js/environments/`:
   `reset(scenario)`, `sampleScenario(random)`, `observe(state)` and
   `step(state, actions)`. The sampler supplies a fresh live starting scenario.
   Return fresh state; observations are one finite index per agent. Step results
   contain `{state, reward, terminated, truncated, events}`. Reward is per agent.
   All actions in this export profile are legal; blocked movement is a transition
   outcome. Preserve Python's action ordering, observation encoding and end flags.
2. Implement its view in `site/js/views/`: `place`, `nudge`, `scene`, and `status`.
   Placement returns numerical state without advancing time. `nudge` maps keyboard
   direction to a requested world position. `scene` converts world states into
   approved asset placements; `status` supplies a short terminal message.
3. Export a versioned directory under `site/data/` containing `manifest.json` and
   `actions.u8`. Use explicit starts and the complete environment configuration.
4. Add one definition to `site/js/demos.js`, with ID, name, task description,
   manifest URL, environment/view factories, and `playback` settings
   (`tickSeconds`, `terminalSeconds`, `epsilon`). The selector appears when
   more than one demo exists. Switching creates a fresh session; only one runs.

The manifest's `art` object belongs to that demo's view. Asset IDs resolve through
`site/assets/manifest.json`; do not bind policy slots to animal filenames.

### Scene contract

```javascript
{
  bounds: [left, bottom, right, top],
  sprites: [{
    id: 'agent-0', asset: 'orbit', agent: 0,
    position: [x, y], width: 1, height: 1,
    heading: 0, opacity: 1
  }]
}
```

World coordinates are x-right/y-up. Width and height are in world units; heading
is clockwise radians from north. Omit `agent` for boards and markers. Sprites are
ordered back-to-front. Bounds stay fixed during a recording. Keep sprite IDs,
ordering, artwork and dimensions stable within it; hide markers with opacity rather
than removing them. This lets Canvas and SVG use the same scene output.

The view receives `(previous, current, fraction, headings)` and must not mutate any
input. It owns geometry-specific interpolation, including angular wrapping for a
ring. The optional headings array preserves the last direction while stationary.
Scene pieces use existing SVGs; a different world surface can be another SVG asset
in the approved style. This implementation does not add a general scene graph.

### Session and input

`createSimulation` receives environment, policy, view, scenarios, fixed tick
interval and stay-action index. It owns state, episode timing and queued placements.
`advance(seconds)` runs fixed ticks; `scene()` returns an interpolated scene.
`tick()` is also usable without a browser for deterministic local checks.

A held piece receives the stay action while its partner keeps acting. Release
commits the legal placement with one stay tick before policy control resumes.
Release snaps immediately to the legal cell, before the next tick, so the piece
does not jump back while waiting for its action. Grid placement allows co-location. Pointer movement adds no
velocity. A terminal world freezes immediately; a held gesture must end before
advancing to the next scenario. Hidden pages suspend time without catch-up.

Pause and Reset are HTML controls. Live Reset samples a new starting state. On the canvas, Enter cycles agent selection,
arrows place the selected agent, and Escape clears selection. Repeated clicking
on overlapping pieces cycles them, starting with the topmost one. The initiating
pointer owns the gesture through release or cancellation. Reduced-motion preference
starts playback paused; Play remains available.

The session retains at most four catch-up ticks per animation frame. Longer stalls
slow visible playback; they do not change individual world transitions. Rendering
is one tick behind logical state during interpolation. Pausing or ending an episode
shows its actual current state.

## Export a frozen grid policy

```sh
JAX_PLATFORMS=cpu python scripts/export_grid_policy.py
```

The default source is the existing `stores/coop_grid_q_learning/latest/q_values.npz`.
For a newly saved independent policy:

```sh
JAX_PLATFORMS=cpu python scripts/export_grid_policy.py \
  --weights runs/my-run/policy/variables.npz
```

Pass `--sharing shared` for a one-slot shared table. Export expands it into one row
per agent. This recipe uses the fixed grid configuration and curated starts; a new
environment needs its own export recipe. It also refreshes the small Python
transition fixtures for the local JavaScript check.

The browser array is row-major `[agent, observation]`, one unsigned action byte
per entry. Greedy ties choose the lowest action index. The grid array is 8,750 bytes.
Full Q-values, training metadata and evaluation reports are not published with it.
Use a new revision directory when publishing a different environment or policy;
point its demo definition to the new manifest.

## Recorded trajectories and animated SVG

The [example grid animation](previews/grid-playback.svg) and its
[recorded trajectory](previews/grid-playback.svg.json) are generated by this path.

Live simulation chooses actions from a frozen policy. A recording stores the
already-computed states and transitions. Replaying it does not call a policy or
an environment transition.

Generate a grid clip and its trajectory sidecar:

```sh
node scripts/export_gallery.mjs grid /tmp/grid.svg 0
```

The final argument selects one of the six curated starts, numbered 0–5. This writes
`/tmp/grid.svg` and `/tmp/grid.svg.json`. To render that saved trajectory again:

```sh
node scripts/export_gallery.mjs grid /tmp/grid-replay.svg 0 /tmp/grid.svg.json
```

With a trajectory argument, the exporter uses its stored environment configuration
and artwork with the selected demo's view. It does not load policy bytes or call
`step`. Use the matching demo ID. Recording metadata includes format/version, demo,
environment, artwork, fixed tick interval and frames. The first frame holds the
initial state; later frames hold state, actions, rewards, events and end flags.

`recordTrajectory` runs a bounded episode without a DOM. `createReplay` produces a
scene at any time from those recorded frames, independent of wall-clock speed.
`exportAnimatedSVG` samples that scene at 30 fps plus exact logical tick boundaries,
then writes native SVG transform/opacity animations. Width, height and fps can be
specified through the JavaScript function. The clip holds its final pose for
600 ms and loops. Its timeline starts movement immediately; live playback waits
for the first simulation tick and uses a one-tick display interpolation delay.

The SVG embeds the approved vector paths once and references them internally.
There is no script, policy, network request or interaction in the result. It remains
scalable and can be used as an image. Viewers without SVG animation support show
the initial pose. Reduced-motion behaviour for embedded clips belongs to the
embedding page; the exported animation has no controls. Keep trajectory JSON as
the source for future exports rather than treating SVG animation values as data.

## Local checks

```sh
node --test tests/browser/runtime.mjs
python -m pytest tests/test_gallery_export.py -q
python tests/browser/smoke.py http://localhost:8876/
```

The first command checks Python transition fixtures, clock independence, placement
semantics, curated trajectories and script-free SVG output. The Python test compares
every exported greedy choice with the saved Q-table. The last command uses an
installed Playwright/Chromium to check real input, selection, reset, switching and
responsive layout against the running server. No training or GPU is needed.
