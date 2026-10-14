# JavaScript runtime audit and redesign

14 October 2026. Scope: the existing grid and the proposed docking and orbit
simulations. This is an implementation specification, not a claim that the new
runtime or environments exist. It refines [Simulation gallery](simulation-gallery.md)
and takes precedence over the broader [Browser handoff](browser-handoff.md)
proposal for this gallery. Python package APIs, public documentation and approved
artwork are unchanged.

## Conclusion

Keep the static delivery stack. Replace the simulation core rather than splitting
the current globals into several files. The useful existing pieces are the saved
policy, SVG assets, basic Canvas drawing, relative navigation and Pages assembly.
The motion loop and pointer handling should not become the template for new demos.

The redesign has four requirements:

1. A deterministic environment transition agrees with the Python training task.
2. Human placement is explicit and separate from the learned action and transition.
3. Drawing and input use the same projection, without modifying world state.
4. Each demo supplies its own geometry and observation encoding to a small shared
   simulation shell. Only one demo runs at a time.

No new Python abstractions, neural inference engine, physics engine, bundler,
framework, worker, plugin registry or server is required for this scope.

## Audit coverage and evidence

Inspected the complete browser module, HTML and CSS; asset metadata; policy JSON;
Python environment, model, agent, storage and experiment entry point; export tests;
and Pages workflow. The following references identify the relevant source sections:

| Source | Relevant responsibility |
| --- | --- |
| [`site/js/gallery.js`](../site/js/gallery.js) | Loading; `stateId`/`chooseAction`; `updatePolicy`/`updatePhysics`/`updateEnvState`; projection; drawing; pointer handlers; animation |
| [`site/index.html`](../site/index.html), [`gallery.css`](../site/css/gallery.css), [`nav.css`](../site/css/nav.css) | Canvas-only interaction, fixed header offsets, navigation |
| [`assets/manifest.json`](../site/assets/manifest.json) | Approved geometry, anchors, facing and scene asset paths |
| [`coop_grid.py`](../src/marlax/envs/coop_grid.py) | Authoritative state, observation, transition, reward and end flags |
| [`policy.py`](../src/marlax/agents/policy.py), [`data.py`](../src/marlax/core/data.py) | Parameter sharing, masks, finite observation indices and greedy execution |
| [`storage.py`](../src/marlax/storage.py), [`run.py`](../experiments/coop_grid_td/run.py) | Current NPZ/JSON policy format; experiment config; no browser export command |
| [`test_gallery_export.py`](../tests/test_gallery_export.py) | Payload shape and scenario bounds only |
| [`pages.yml`](../.github/workflows/pages.yml) | Sphinx build and static copy into `/gallery/`; no browser tests |

The first audit's controlled-clock measurements and pointer checks are recorded in
[Simulation gallery](simulation-gallery.md#local-checks-performed). Additional
Chromium probes in this audit established:

| Probe | Observed result | Consequence |
| --- | --- | --- |
| Put an agent at `(0,0)`, click the far-left canvas margin at its screen y | Agent selected | `pointerToGrid` clamps before hit testing, so clicks outside the board can select an edge piece |
| Right-click an agent | Drag begins | Handler does not restrict the initiating mouse button |
| During a mouse drag, dispatch a move with another pointer ID | Held agent moves to the foreign pointer | Capture alone does not establish ownership; handlers ignore `pointerId` |
| Dispatch a release from that other pointer | Original drag ends | Multi-pointer ownership must cover release as well as movement |
| Inspect canvas focusability and handlers | No `tabindex` or keyboard controls | Pointer-only interaction |

Foreign-pointer probes used dispatched PointerEvents; these establish the handler
behaviour, not a claim of testing physical multi-touch hardware. No new training,
Safari/Firefox testing, network performance benchmark or GPU work was performed.
The earlier frame-rate test advances update functions with synthetic time; it is
not a real-device frame-rate benchmark. No production code was modified by probes.

## Findings, ordered by dependency

### 1. There is no independent environment state

`agents` mixes policy identity, image objects, position, rounded cell, velocity,
heading and speed. Other task state lives in globals (`active`, `targetId`), while
step count is absent. Rendering mutates heading; pointer handlers mutate both
continuous position and discrete cell. It is therefore impossible to isolate a
world transition from the browser and compare it directly with Python.

Retain only numerical task state in an environment: positions/angles, target or
phase, and steps. Keep images, selection, animation endpoints and visual heading
outside it. Policy tables are read-only. Never derive observations from displayed
positions or from sprite geometry.

### 2. Smoothing currently changes the task

The learned grid action is one clipped integer move. The browser instead adds
velocity toward a neighbouring cell, damps that velocity each rendered frame, and
rounds the resulting position back into a cell. Pointer repulsion adds another
untrained force. Changing the display refresh rate changes the observed states,
actions and episode progress. Updating `dt` alone cannot repair this semantic gap.

Remove this velocity system for the grid. Smooth only the drawing between two
logical cell centres. Docking uses its explicitly trained fixed-distance action
rule; orbit uses its explicitly trained angular step. Neither needs momentum.

### 3. Episode semantics are incomplete

The current `updateEnvState` handles centre activation and collection, but does not
implement wrong-target failure, truncation or rewards. Collection is checked once
per rendered frame and policy actions continue during the frame-counted hold.

Port the complete grid transition in Python's order: simultaneous clipped moves;
activation; collection/split/wrong-zone checks using the updated phase; additive
reward; step increment; termination and time-limit truncation. Both end flags may
be true on a terminal last step. Preserve both rather than making them exclusive.
Split occupancy incurs a penalty but does not terminate. World rewards remain
internal; they do not require public statistics or charts.

### 4. Input conversion hides geometry mistakes

Current pointer conversion clamps before selecting. The selection radius is 0.85
cells regardless of the rendered silhouette, mouse versus touch, or display scale.
With a new arena or ring, a grid-cell radius is not meaningful. The handler also
jumps the agent anchor to the pointer rather than retaining the grab offset.

Convert pointer coordinates to unbounded world coordinates. Hit-test displayed
anchors in CSS pixels before projecting a placement into legal world space. Use a
small screen-space hit radius tied to rendered size, with a minimum touch target;
choose overlapping candidates by draw order, then cycle repeated selections.
Record the pointer-to-anchor offset on pickup so the piece does not jump.

### 5. Rendering assumes every world is a square lattice

`board.cell`, `gridToCanvas`, `drawGrid`, `drawMarkers` and `drawMouse` are coupled to
five-cell geometry and two particular mice. The approved circuit's 56-unit inset
and 400-unit playable span are correct asset-specific facts, not universal world
projection parameters. Asset indexes in an array are implicit identities.

Keep grid board alignment in the grid view. Resolve images by asset ID, use their
anchors and facing metadata, and keep body collision radius independent of artwork
extent (including monkey tails). Share Canvas projection and sprite drawing, but
let each demo build its scene and interpolate its own geometry.

### 6. Demo lifecycle is currently just page lifecycle

Top-level fetch/decode is followed by one permanent animation chain and listeners.
There is no switch/dispose contract. Simply adding more calls would create duplicate
loops or stale input references. The CSS fixes the world below a 72/112-pixel header;
adding demo controls there would cover or shrink the canvas incorrectly.

Keep one application-owned canvas, input controller and animation chain. Switch the
active session, not the whole application. Use a normal layout with header,
compact demo controls and the remaining canvas area; size from the actual canvas
container. Keep the existing documentation and assets navigation.

### 7. Browser export is disconnected from current persistence

The live page reads a historical monolithic Q JSON. Current `save_policy` writes
`variables.npz` plus a manifest, while experiment configuration holds the world
parameters. A saved policy's `environment_spec` alone does not contain enough
information to recreate a world. The experiment command explicitly never writes
the gallery. A release needs a deliberate export recipe joining policy, environment
configuration and explicit starting states.

`Agent` supports shared as well as independent parameter slots; the current browser
indexes slots directly by agent ID. Also, a greedy action table is valid only if
legality is constant or determined by the exported observation index. These three
demos should keep all actions selectable: walls/collisions are transition outcomes,
not hidden state-dependent masks. No runtime Q-table masking is then necessary.

## Target ownership and module boundaries

```text
site/
  package.json                 private, type=module; local check command only
  js/
    gallery.js                 DOM, bundle loading, active session, one RAF chain
    simulation.js              fixed tick, episode state, pause/reset, interventions
    input.js                   owned pointer, hit selection, queued interaction
    policy.js                  read-only Uint8Array lookup
    render.js                  Canvas projection and shared asset drawing
    environments/
      grid.js                  reset/observe/step; no DOM or rendering
      docking.js               added with its own Python environment and policy
      orbit.js                 optional
    views/
      grid.js                  circuit layout, placement, scene, interpolation
      docking.js               arena layout, placement, scene, interpolation
      orbit.js                 ring layout, placement, scene, interpolation
  data/<demo>/<revision>/
    manifest.json
    actions.u8
```

The private package manifest makes `.js` modules explicit for small local Node
checks. It needs no runtime dependencies or lockfile without dependencies; Pages
continues copying static files without an npm install or build. Tests and reference
fixtures live outside `site/` so publication does not ship them. The module names
are boundaries, not a requirement to add empty docking/orbit files immediately.

| Owner | State owned | Must not own |
| --- | --- | --- |
| Environment | Pure task state; configured transition and observation functions | Pointer IDs, images, clocks, selection, DOM |
| Policy | Immutable action bytes; observation count | World mutation, reward, artwork |
| Simulation session | Current/previous state, accumulator, episode result, pending placement, held agent | Canvas coordinates or SVG paths |
| Input | Captured pointer ID, grab offset, selected agent, latest requested position | Task phase or policy execution |
| Demo view | Geometry-specific placement and scene projection | Learning or episode advancement |
| Renderer | Images and drawing context | Mutation of numerical task state |
| Gallery application | Current session, controls, loaded bundle, one RAF loop | Environment-specific collision/target rules |

Environment methods, closed over configuration:

```javascript
const state = environment.reset(scenario);
const indices = environment.observe(state); // one finite index per agent
const result = environment.step(state, jointActions);
// result: {state, reward, terminated, truncated, events}
```

`reward` is a per-agent array, matching Python's TimeStep. `events` contains only
small task facts used by the view (for example activated/collected/wrong target).
A no-input sequence uses exactly this API. Do not pass `heldAgentId` into `step`.

View methods, closed over layout and placement configuration:

```javascript
const placedState = view.place(state, agentId, requestedWorldPoint);
const scene = view.scene(previousState, state, fraction);
renderer.draw(scene, interactionOverlay);
```

`place` changes only the selected agent's position/angle and returns fresh numerical
state. It does not increment time, award reward, activate a phase or terminate an
episode. The view is DOM-free too, so placement/interpolation can be checked locally.
The scene contains board/track primitives, marker positions and piece IDs, world
positions, facing and artwork IDs. This is a small drawing description, not a scene
graph, ECS or public graphics API. Keep snapshots independent so in-place array edits
do not corrupt interpolation endpoints.

## One tick: exact interaction ordering

All three action sets have a declared `stay` action. Holding therefore belongs in
the session's action selection, not in an alternate environment transition.

1. Take the latest queued placement and apply `view.place` to current state.
2. Compute all observations from that one placed state.
3. Look up every frozen action.
4. Replace the held agent's action with `stay`. A release pending this tick also
   receives `stay` once, so it actually occupies its drop location for a transition.
5. Call the ordinary `environment.step` once. This evaluates task events and advances
   episode time while the partner continues acting.
6. Save the result, consume a pending release, and set interpolation endpoints.
   A placed piece jumps to the placement; it never sweeps visually across the world.
7. If terminated or truncated, stop further ticks and enter terminal presentation.

This explicitly resolves a gap in the first design: observing a dropped position
and immediately moving away could skip activation or a terminal location. The
single stay tick makes the drop part of normal transition semantics. Once it ends,
policy control resumes. Interventions are human moves, not claims that the frozen
policy itself chose those actions.

While held, accept the latest pointer location each tick; do not replay every DOM
move event. The held actor still participates in collisions, meeting and terminal
rules. Time limits continue while running. Target identity and phase never reset
because a user drags. No drag impulse or pointer repulsion exists.

While paused, placement updates numerical position for immediate display but does
not evaluate rewards/events or increment steps. Keep a pending stay tick for resume.
Do not offer further dragging on a terminal world. If termination occurs during a
hold, freeze the terminal positions immediately and stop accepting placement. Keep
capture until release/cancel, then clear ownership before the scenario changes. This
replaces the earlier ambiguous instruction to keep dragging a completed episode.

## Clock and lifecycle rules

Use integer tick counts with a fixed configured interval: grid 180 ms for
presentation, docking/orbit 50 ms as their proposed physical timestep. Grid timing
controls playback speed only; continuous task speed is tied to its trained `dt`.

Accumulate elapsed time and subtract the interval per tick, preserving remainder.
Bound catch-up to four ticks in one frame; after a longer stall discard excess whole
ticks and retain the fractional remainder. This deliberately slows playback under
severe stalls rather than jumping through many unseen episodes. It must not change
the transition sequence per tick. Ordinary 30–144 Hz rendering should give the same
sequence and elapsed simulation time; hidden-page time is never accumulated.

Interpolate previous/current states with the remaining fraction (a one-tick visual
lag). At reset or placement set the affected endpoints equal. At termination display
the actual terminal state, not a partial interpolation. Hold it for 600 ms of visible,
unpaused presentation time, then move to the next start. No additional policy steps
occur during this hold. This time is display-only, unrelated to training `dt`.

| Event | Session behaviour |
| --- | --- |
| Pause | Stop tick and terminal timers; keep selected scenario and state |
| Play | Reset wall-clock baseline; resume without accumulating paused time |
| Reset | End gesture; reset current scenario, tick count, phase, overlays and clocks; preserve user's paused/running choice |
| Hidden page | Suspend timers; cancel held gesture at last committed placement; keep user pause choice |
| Visible page | Resume only if user had left it playing; reset wall-clock baseline |
| Pointer cancel/lost capture | End hold at last committed legal position; clear uncommitted preview |
| Demo switch | End gesture, suspend old session, load requested bundle; start its first scenario with fresh clocks |

Attach event listeners once to the application input controller. During a demo
load, disable selection controls until that load completes; do not allow competing
loads to activate stale sessions. Pause the old session while loading. This small
serial lifecycle is sufficient here; no generalized asynchronous task manager.
Cache decoded approved images by URL across switches. Load only the selected demo's
policy; no need to fetch or run all three at startup.

## Coordinate, drawing and selection contract

Use x-right/y-up numerical coordinates. Canvas calculations use CSS pixels; scale
the backing store once for device pixel ratio. Given world centre `(cx, cy)`, screen
centre `(sx, sy)` and uniform scale `k`:

```text
screenX = sx + (worldX - cx) * k
screenY = sy - (worldY - cy) * k
worldX  = cx + (screenX - sx) / k
worldY  = cy - (screenY - sy) / k
```

Hit testing uses that inverse without clamping. Placement then applies the demo's
constraint. For a north-facing SVG, motion `(dx,dy)` yields Canvas rotation
`atan2(dx,dy)`. Rotation should use world motion or track tangent, not a fake physical
velocity. Preserve the previous visual heading at rest. Artwork metadata determines
the anchor; it does not determine the environment's collision body.

Draw in order: board/track, task markers, pieces, selection and valid-placement
indicator. Keep white opaque piece interiors and the approved black/dark-red palette.
Place goal labels outside docking markers. Do not introduce new animal art.

The grid view owns the Circuit SVG's `56,56` origin and 80-unit cell pitch. Docking
has a bounded continuous square. Orbit maps angle to `(r*cos(theta),r*sin(theta))`.
Interpolate orbit angles along the wrapped shortest arc, not a straight chord
between Cartesian endpoints; a wrap through zero must not rotate almost a full turn.
Continuous docking uses linear position interpolation, grid uses cell interpolation.

Hit-test the currently drawn anchors, not the next logical state. Selection does
not rewind simulation; after pickup, the placed agent follows the pointer with its
grab offset. Record the active pointer ID and accept only its move/up/cancel events.
Start mouse interactions only on the primary button. A second pointer cannot replace
or release the first. Empty presses have no simulation effect.

Keep keyboard controls explicit: focusable canvas, agent selection with a key or
small labelled control, arrows to place the selected agent, Escape to end selection.
Grid arrows move one cell, docking arrows a fixed placement increment, orbit
left/right one angular increment. Route these through the same placement/stay-tick
path. HTML buttons provide Pause/Play and Reset; do not trap browser scrolling or
keys when the canvas is unfocused. Provide brief task text outside the canvas and
announce only selection and terminal changes, not every animation frame.

## Environment-specific readiness

| Requirement | Grid | Docking proposal | Orbit proposal |
| --- | --- | --- | --- |
| Authoritative state | Integer positions, target ID, active, steps | Float positions, fixed assigned goals, steps | Angles, meeting phase, steps |
| Actions | Existing five; stay=0 | Stay + eight normalized directions | Stay, CCW, CW; freeze order before training |
| Body interaction | Co-location allowed | Two circles; collision rule below | Co-location allowed |
| Placement | Nearest in-bounds cell | Clamp body centre to arena; reject overlap | Project onto circle by angle |
| Observation | Existing shared joint index | Quantized local displacements | Quantized own/relative angles and phase |
| Policy available | Yes | No | No |
| Geometry-specific interpolation | Cell centres | Continuous positions | Wrapped angles |

### Docking needs a complete task specification before export

The first sketch specifies 9×9 goal-displacement bins and 5×5 partner-displacement
bins, but not their ranges, clipping, boundary rules or flattening order. Those
omissions can make identically shaped tables choose different actions across Python
and JavaScript. Record edges, order and endpoint inclusion in the environment
version; use identical encoder formulas. All actions remain legal, even when their
outcome is blocked, so no unobserved state-dependent mask invalidates the action table.

Fixed assigned goals mean absolute location is recoverable from exact goal-relative
position and agent identity, but coarse bins still alias wall distance and collision
clearance. The frozen table is a plausible small baseline, not a guarantee of robust
continuous navigation. Variable goals, obstacles and velocities would require more
state; do not silently introduce them in the demo layout.

For a tick, compute both candidate displacements from the same state and clamp body
centres to the inset arena. Let initial relative position be `r` and relative candidate
displacement be `d`. The closest synchronous approach over the tick is at
`clamp(-dot(r,d)/dot(d,d), 0, 1)`; if `d` is zero use the initial separation. Reject
both candidate moves when separation is less than the sum of body radii. Equality
is allowed. This detects crossing paths without an iterative physics engine.
Define whether blocked moves incur one team collision penalty (recommended), and
compute progress reward from accepted positions. A held body has zero displacement
and still participates. Check terminal docking after accepted movement.

Simplify placement from the first sketch: clamp to the legal centre bounds; if the
candidate overlaps the other body, retain the last legal position. The placement
indicator stays there. Do not promise a globally nearest valid point, which needs
additional geometry at corners and can cause surprising jumps. Recheck each tick
against the partner's current position. Dragging itself teleports; it does not sweep
through intervening bodies or earn distance-progress reward for the teleport.

Before training, pin arena size, body/goal radius, speed, timestep, time limit, fixed
goal assignments, reward values and bin definitions. Then a short local training and
relocation check decides whether this tabular task is good enough to publish. Keep
this uncertainty out of the public demo; do not add new models pre-emptively.

### Orbit is smaller but not a grid with curved rendering

Use radians in `[0, 2*pi)`. Define positive motion as counterclockwise and a wrapped
signed difference in `[-pi, pi)`. Freeze those conventions and the bin origin.
A suitable encoder is own-angle bin + `24 * relative-angle bin + 576 * phase`,
with 24 half-open bins per angle; total 1,152 indices. Specify that relative angle
means partner minus self. Handle negative remainder explicitly in both languages.

Choose fixed angular step smaller than the meeting/target tolerance so agents cannot
jump across the entire completion region. Check both agents against the current
phase destination after movement. Apply progress shaping against the pre-step phase
so switching destinations does not accidentally create a large negative reward.
The episode can advance to the target phase only once.

At the exact track centre the pointer has no angle: retain the last valid placement.
Otherwise project with `atan2`. This is an ordinary ring placement rule, not generic
error recovery. Dropping does not change phase. Wrapped interpolation and tangent
heading are responsibilities of the orbit view, independent of policy inference.

### Numerical agreement

Grid transitions and observations should match exactly. JavaScript Number arithmetic
is double precision while JAX generally uses float32 here. For docking/orbit, merely
writing the same formulas is insufficient near bin, collision or terminal thresholds.
Specify float32 state storage and rounding points in the reference environment;
mirror them with Float32Array/Math.fround where required in its JS port. Compare
continuous state with an explicit tolerance, but require equal encoded observations,
actions and event flags on selected boundary fixtures. Do not hide an observation
mismatch behind a positional tolerance or claim unlimited bitwise equivalence for
trigonometric functions. Freeze the encoder contract before collecting a policy.

## Minimal browser artifact

Use the current `marlax.browser` version-1 proposal as a new export format; replace
the historical JSON deliberately, without a legacy loader. Export deterministic
greedy actions only. For this gallery, expand shared parameter slots into one row
per agent at export time. The browser then always indexes `agent * S + observation`;
this costs negligible space and avoids exposing model sharing in the runtime.

A release manifest needs:

- Format/version and demo revision.
- Environment ID/version, complete numerical configuration, observation schema,
  action ordering, stay-action index, and coordinate convention.
- Policy path, unsigned-byte dtype, shape `[agents, observations]`, row-major order
  and lowest-index tie rule.
- Explicit initial states, not seeds; display tick interval and approved asset IDs.

For all-action-legal policies, export `argmax` from the original NPZ values and
compare every exported action before publishing. Grid uses 8,750 bytes, docking's
proposed table 4,050 bytes, orbit 2,304 bytes, excluding their manifests. Do not
ship Q inspection data, reference fixtures, training history or evaluation reports.
Source model hashes and release provenance can stay with the export recipe locally.

Resolve paths against the fetched manifest URL. Use revision directories so a
changed manifest does not point to overwritten policy bytes from another release.
The exporter and runtime agree on one schema; no broad compatibility adapter or
runtime recovery framework. The saved historical grid NPZ may be converted once
by an explicit recipe. Future recipes consume `load_policy` plus experiment config;
checkpoint leaf arrays are not a browser format.

## Small implementation slices and acceptance checks

| Slice | Deliverable | Small local evidence before proceeding |
| --- | --- | --- |
| 1. Exact grid | Pure grid state/observe/step and action lookup | Joint-move fixtures: clipping, activation, collection, split, wrong target, timeout, and terminal+timeout together; compare per-agent rewards too |
| 2. Session | Fixed tick, terminal hold, pause/reset | Equal state sequences for equal ticks across frame intervals; no hidden backlog or post-terminal policy steps |
| 3. Interaction/view | Approved grid drawing and explicit placement | Drop-on-centre/target, partner continues, pause/drop/resume, overlapping selection, pointer ownership/cancel, margin click, keyboard and resize |
| 4. Release | Compact artifact, small selector, module boundaries | Every exported action matches source; switch/reset leaves one active loop; assets and manifests resolve under `/marlax/gallery/` |
| 5. Second demo | Docking reference environment, port and frozen policy | A few collision/bin-boundary parity fixtures plus representative start and relocation playback |
| 6. Optional third | Orbit reference environment, port and policy | Angle-wrap and phase-transition fixtures; constrained dragging |

Use a small Node check for DOM-free transitions/clock and one manual or Playwright
browser smoke pass for interaction. Add checks only when the corresponding slice is
implemented. No exhaustive screenshot suite, large benchmark run, or tests in Pages
CI. Use CPU for these browser checks; GPU training is unrelated to JavaScript
readiness. The existing workflow can keep publishing the static directory as-is.

## Corrections to earlier design notes

- Keep `environment.step(state, actions)` pure; suppress the held action in the
  session rather than adding pointer knowledge to the environment.
- Define a one-tick stay on release, paused placement, and terminal-during-drag
  behaviour explicitly.
- Separate geometry-specific views from the shared renderer, especially orbit
  interpolation and docking placement.
- Replace docking's nearest-valid-point promise with clamp-and-reject placement.
- Treat docking/orbit observation encoding and numerical parity as task design
  prerequisites, not details that a generic manifest can infer.
- The shipped bundle contains inference inputs only. Earlier optional Q/evaluation/
  reference files and method/scenario selectors are outside this gallery's scope.
