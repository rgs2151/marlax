# Interactive simulation gallery

**Scope update:** Paired docking and orbit rendezvous are rejected, unimplemented
proposals. Their details below are historical, not the current demo plan. See
[Demo candidates](demo-candidates.md) for the researched shortlist and the shift
toward recorded demonstrations.

Design and local JavaScript audit, 14 October 2026. No runtime or environment
implementation is included in this change. This document narrows the earlier
[browser handoff](browser-handoff.md) proposal to two or three interactive demos.
The Python package and public documentation stay unchanged.
The follow-up [JavaScript redesign audit](javascript-redesign.md) specifies the
interfaces, tick ordering, geometry and handoff in more detail and takes precedence
where it refines this initial sketch.

## Current condition

The gallery loads, renders the approved SVGs, reads a frozen policy, and accepts
mouse dragging. It is a useful visual prototype, but its dynamics do not reproduce
the trained environment. Fix that before copying it into additional demos.

| Area | Finding | Required change |
| --- | --- | --- |
| Setup | One native ES module, Canvas 2D, static JSON and SVGs; no build dependencies | Keep this setup; split responsibilities into small modules |
| Policy | Joint-state indexing and greedy tie-breaking match Python | Preserve these conventions |
| Motion | `updatePolicy` applies acceleration; `updatePhysics` damps velocity by 0.84 every frame and rounds positions into observations | Run the actual discrete grid transition; interpolate only for drawing |
| Clock | Decision timer discards remainder; success delay counts rendered frames | Fixed simulation ticks, elapsed-time presentation delays |
| Episodes | Centre activation and collection exist; wrong-target termination, split penalty and the 48-step limit are absent | Port the full transition and terminal semantics |
| Dragging | Release keeps a fractional position and nonzero velocity; pointer proximity also pushes other agents | Explicit placement with no pointer forces or release momentum |
| Selection | Coincident agents always select agent 0, although agent 1 is drawn on top | Select topmost first, cycle overlapping candidates on repeated selection |
| Pointer lifecycle | Empty presses are not captured; releasing outside can leave pointer state active | Only start interactions on a piece; capture and track that pointer through release/cancel |
| Coordinates | Logical y is drawn downward, mirroring the Python view | Use y-up world coordinates and one invertible canvas projection |
| Lifecycle | One global world and animation loop; no pause/reset controls | One active demo, one loop, explicit pause/reset and visibility handling |
| Artifact | A 760,664-byte JSON includes full Q-values and training diagnostics | Publish only inference data and scene configuration |
| Checks | Existing export tests inspect payload shape, not JavaScript dynamics | A small local parity check plus pointer smoke check |

Evidence: [`gallery.js`](../site/js/gallery.js),
[`CooperativeGrid`](../src/marlax/envs/coop_grid.py),
[`export tests`](../tests/test_gallery_export.py).

### Local checks performed

Chromium loaded the page with no JavaScript exceptions. Desktop and narrow-screen
renders were inspected. A temporary Playwright harness called the existing update
functions with controlled frame intervals; it did not modify production code.

| Render rate | Scenario advances in 60 simulated seconds |
| --- | ---: |
| 30 fps | 20 |
| 60 fps | 15 |
| 120 fps | 9 |
| 144 fps | 8 |

These are a clock-dependence diagnostic, not trained-policy performance results.
The same simulation should follow the same state trajectory at every render rate.

Pointer checks confirmed that dropping at approximately `(1.35, 1.25)` retained
velocity `(0.70, 2.50)`, selecting overlapping pieces repeatedly chose agent 0,
and releasing an empty-space press over the header left `pointerDown` true.
Injecting an active wrong-target state did not terminate the episode.

The exported Q-values agree with the saved NPZ to about `5.1e-6`; all 8,750 greedy
choices agree. The immediate problem is the browser transition, not a damaged
policy file. Touch and cross-browser behaviour have not been verified.

## Small JavaScript structure

Keep native JavaScript modules, Canvas 2D, the approved SVG library, and static
GitHub Pages hosting. These demos do not require a framework, bundler, physics
engine, server, GPU, or browser training.

```text
site/js/
  gallery.js           DOM, demo selection, loading and teardown
  simulation.js        fixed clock, policy decisions, episode lifecycle
  input.js             pointer capture, selection and placement requests
  policy.js            frozen greedy action lookup
  render.js            projection, SVG drawing and interpolation
  environments/
    grid.js            exact cooperative-grid transition
    docking.js         proposed second environment
    orbit.js           optional third environment
  views/               per-demo layout, placement and interpolation
```

Use explicit imports and a short array of demo definitions. No plugin discovery
or general-purpose environment framework. Each environment supplies `reset`,
`observe` and `step`. A separate demo view supplies placement, scene layout and
interpolation.

Proposed call boundary:

```javascript
state = environment.reset(scenario);
state = view.place(state, agentId, worldPoint);
const observations = environment.observe(state);
const actions = policy.act(observations);
actions[agentId] = stayAction; // session holds the placed agent for this tick
const result = environment.step(state, actions);
state = result.state;
renderer.draw(view.scene(previousState, state, interpolation), selection);
```

`step` returns `{state, reward, terminated, truncated, events}` with per-agent
rewards. The timestep belongs to the environment configuration, not the display frame. With no held agent,
transitions must match Python exactly. A held agent is an explicit interactive
intervention: its move is suppressed while task rules and the other agent continue.
Compute every policy action from the same pre-step state.

Start the grid with one logical move per 180 ms. Keep a clock accumulator and its
remainder. Rendering interpolates between completed states; interpolated positions
never become observations. Use real elapsed time for the brief terminal display,
then reset to the next curated start. Pause when the page is hidden and reset the
wall-clock baseline on return, without running a hidden-page backlog.

## Interaction contract

- Drag a piece while the other agent continues acting. Empty board presses do nothing.
- Show a dark-red selection outline and the valid drop location.
- Queue placement for the next simulation tick. On the grid, round to the nearest
  in-bounds cell; the visual drag preview may follow the pointer freely.
- On release, commit the placement and resume policy control with no inherited
  pointer velocity. Grid co-location remains legal, as required by the task.
- Keep target identity, activation phase and episode step count. Dragging changes
  position, not the task. Normal success/failure rules still apply.
- If a terminal event occurs during dragging, freeze task positions and stop
  accepting placements; wait for release/cancel before advancing the scenario.
- Track one captured pointer. Release and cancellation both end the hold. Repeated
  selection at an overlap cycles pieces, starting with the visually topmost one.
- Provide Pause/Play and Reset. Paused placement remains paused. Provide a simple
  keyboard equivalent: select a piece and move it with arrow keys.

The page needs a short task description, “Drag an agent to move it,” and visible
Documentation / Assets navigation. Select demos with a small row of links or tabs;
only the selected demo runs. Do not add learning controls, parameter panels,
Q-value displays, charts, or a scene editor.

## Proposed environments

The [concept sheet](previews/simulation-gallery.svg) uses existing approved pieces
and markers. Arena boundaries are schematic layouts, not new approved asset sets.

### 1. Cooperative grid — existing anchor

Two mice meet on the central meeting point, revealing two permitted destinations.
They must choose the same destination. Preserve the exact existing task, policy,
Circuit board, target beacons and meeting-point artwork. Dragging demonstrates how
one frozen policy reacts to a changed joint state. First repair the runtime;
no retraining is needed merely to remove its added physics.

### 2. Paired docking — second demo

Two vehicles move freely inside a square arena toward their respective fixed
beacons. They must both dock; crossing paths makes coordination visible. Start
without obstacles, inertia or steering mechanics.

- Continuous 2D positions; nine discrete actions: stay and eight directions.
  Normalize diagonal movement. Use a fixed 20 Hz transition and fixed speed.
- Give each agent its own goal. Both within their goal radii ends the episode;
  a fixed time limit ends an unsuccessful attempt.
- Jointly propose moves, clamp to the arena, then reject both moves if their swept
  paths violate the body separation radius. Use the same calculation in JAX and JS.
- Shared reward combines progress toward both goals, a small step cost, collision
  penalty and joint docking bonus. Set concrete values during the small experiment.
- Quantize each agent's goal displacement into 9 × 9 bins and the other agent's
  relative position into 5 × 5 bins: 2,025 observations per agent. This permits the
  existing tabular model and learning methods with continuous world positions.
- Dragging clamps to the arena and excludes the other body; show the valid
  placement. Reject overlapping placements and keep the last valid position,
  rather than solving for the nearest feasible point. Two goal labels distinguish
  assignments without extra colours.

This requires a new environment and newly trained policy, not reuse of the grid
weights. Quantization aliases states; validate recovery after relocation before
choosing curated starts. No neural-model API changes are needed for this proposal.

### 3. Orbit rendezvous — optional third demo

Two monkeys travel around a circular track. Both first meet at the meeting point;
then both travel to the opposite target beacon. This offers a different geometry
with a very small state and action space.

- Continuous angles; clockwise, stay and counterclockwise actions. Fixed angular
  speed and 20 Hz transitions, with no inertia. Co-location is allowed.
- A phase bit records whether the initial meeting has happened. Use angular
  tolerances for meeting and completion, and a finite episode limit.
- Quantize own angle and relative partner angle into 24 bins each, plus phase:
  1,152 observations per agent. Shared progress reward follows the current phase,
  with meeting/completion bonuses and a small step cost.
- Project dragging onto the track; show the landing point. Releasing at a new angle
  allows both agents to coordinate again from their new positions.

Train this separately using the existing tabular API. Build it only if the grid
and docking demo leave a useful gap; two polished simulations meet the scope.

## Frozen policy handoff

Retain NPZ and its metadata for Python checkpoints. A browser release is a small
inference artifact, not the training checkpoint. For these deterministic tabular
demos, export one unsigned byte per `(agent, observation)` containing the greedy
action. Grid policy data would be 8,750 bytes before compression, excluding its
manifest, rather than the current 760,664-byte JSON. These are file sizes, not
measured download times.

Each demo gets a JSON manifest containing schema version, environment identifier
and transition version, world parameters, coordinate convention, action ordering,
observation encoding, policy shape and binary path, plus curated starting states.
The policy file is row-major by agent then observation; tie-breaking uses the
lowest action index. Resolve artifact paths relative to the manifest URL. Keep
training diagnostics and benchmark results out of browser bundles and public UI.

A small export recipe can read the existing saved model without changing the
package's persistence API. No support for arbitrary neural graphs is needed here.

## Implementation order and minimal verification

1. Separate the current grid transition from rendering and remove frame-dependent
   physics. Verify matching Python/JavaScript state sequences for a handful of
   joint actions covering activation, collection, split, wrong target and timeout.
2. Implement the interaction contract and fixed clock. Check the same tick sequence
   at 30/60/120/144 fps and manually smoke-check dragging, overlapping selection,
   pointer cancellation, narrow screens, pause/reset and demo switching.
3. Export the compact frozen policy and compare every greedy action with the NPZ.
   Keep the six existing grid starts as a quick local playback check.
4. Add docking through the same boundaries, train a small policy locally, and check
   both ordinary starts and a few deliberate relocations. Add orbit only afterward.

Keep these checks local and small; no training benchmarks or test jobs in deployment
CI. This design does not change the existing GitHub publishing workflow.
