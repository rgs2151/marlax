# Frozen policies and the JavaScript library

For the current two-to-three-demo gallery, the
[JavaScript redesign audit](javascript-redesign.md) takes precedence over this
earlier, broader proposal. In particular, use inference-only bundles, live dragging
with a one-tick hold on release, and a small demo selector without method controls.

## Boundary

Python owns learning and reference evaluation. JavaScript owns browser inference,
world stepping, and interactive control. Rendering owns pixels. The handoff is a
versioned directory of data, not serialized Python objects or executable code.

There are three distinct operations:

- **Train:** update weights, tables, optimizer, replay, or a learned dynamics model.
- **Simulate:** choose actions with frozen parameters and advance the environment.
- **Replay:** display an already recorded trajectory without choosing new actions.

The gallery defaults to simulation. Reference replays can be offered separately
and labeled accordingly. Interacting with a frozen policy does not train it.

## Two independently reusable libraries

`marlax` is the Python learning library. `web/runtime` is a JavaScript library
usable without the MARLAX gallery UI. Its first public surface is conceptually:

```javascript
const bundle = await loadBundle("./demos/mice-rendezvous/v1/manifest.json");
const policy = createPolicy(bundle.policy);
const env = createEnvironment(bundle.environment);
const simulation = createSimulation({ env, policy, scenario: bundle.scenarios[0] });

simulation.tick();
const snapshot = simulation.snapshot();
renderer.draw(snapshot, theme);
```

`loadBundle` resolves array/scenario paths relative to the manifest URL, including
when GitHub Pages serves the site under a repository subpath. `createEnvironment`
selects a known runtime module by ID/version. It cannot execute an arbitrary
Python environment merely because its name appears in JSON.

The low-level world port exposes `reset(scenario)`, `observe(state)`, and
`step(state, jointActions)`. `step` returns the resulting state, observations,
reward vector, and termination/truncation flags, with the same meaning as Python.
Methods use immutable numerical state or explicit replacement values. Simulation
owns state and carry; the renderer receives snapshots only.

For the first grid port, retain these concrete rules: clip movement at boundaries;
allow animals to share a cell; activate only when both occupy the center; collect
only when both occupy the same selected target; end in failure when any active
animal reaches an unselected edge zone. Occupying both selected targets separately
incurs the split penalty but does not end the episode. Reward components add,
including the step cost on terminal steps. Apply the time limit after movement.
These rules belong in the Python environment and its explicit JavaScript port,
never in the renderer.

## Bundle layout

```text
artifacts/mice-rendezvous/v1/
  manifest.json
  actions.u8                    # compact greedy table
  q.f32                         # optional Q inspection data
  scenarios.json                # explicit initial states, not training seeds
  evaluation.json               # summary tied to these exported parameters
  reference.json                # a few expected trajectories for parity
```

Use a distinct format ID `marlax.browser` and format version `1`. This is a new
contract, not version 1 of the current gallery's `schema_version: 2` JSON. The
prototype payload is re-exported deliberately; do not add a legacy loader.

Illustrative manifest for the first migrated demo:

```json
{
  "format": "marlax.browser",
  "version": 1,
  "id": "mice-rendezvous",
  "revision": "v1",
  "source_run": "coop_grid_td/example-run",
  "environment": {
    "id": "coop_grid",
    "version": 1,
    "agent_ids": ["agent_0", "agent_1"],
    "config": {
      "size": 5,
      "time_limit": 48,
      "limit_kind": "truncation",
      "activation_reward": 0.2,
      "target_reward": 1.0,
      "step_reward": -0.01,
      "wrong_target_reward": -1.0,
      "split_target_reward": -0.4,
      "center": [2, 2],
      "zones": [[2, 4], [4, 2], [2, 0], [0, 2]],
      "targets": [[[2, 4], [4, 2]], [[4, 2], [2, 0]], [[2, 0], [0, 2]], [[2, 4], [0, 2]], [[2, 4], [2, 0]], [[4, 2], [0, 2]]]
    },
    "coordinates": "x_right_y_up",
    "actions": ["stay", "south", "north", "west", "east"],
    "moves": [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]],
    "observation_schema": "coop_grid_joint_index_v1"
  },
  "policy": {
    "kind": "greedy_table",
    "version": 1,
    "agent_to_slot": [0, 1],
    "observation_schema": "coop_grid_joint_index_v1",
    "actions": {
      "file": "actions.u8",
      "dtype": "uint8",
      "shape": [2, 4375],
      "order": "C"
    },
    "q_values": {
      "file": "q.f32",
      "dtype": "float32",
      "byte_order": "little",
      "shape": [2, 4375, 5],
      "order": "C"
    },
    "tie_break": "lowest_legal_action_index"
  },
  "scenarios": "scenarios.json",
  "evaluation": "evaluation.json",
  "reference": "reference.json"
}
```

The run ID is illustrative, not a claim that this artifact exists. `q_values` is
optional; omit it entirely for the smallest runtime bundle. Production exports
also record source commit and file SHA-256 values in provenance so evaluation
and the published arrays have an identifiable relationship.

### Array layout and encoding

The greedy table has `[policy_slot, observation_index]` entries. Its flat offset
is `slot * S + observation_index`. Q inspection uses `[slot, S, U]` with offset
`(slot * S + observation_index) * U + action`. Export raw bytes in the declared
order; JavaScript reads unsigned bytes or little-endian float32 values. Do not
round Q-values through decimal JSON to save space.

For the existing two-agent, size-5 observation:

```text
cell_i = y_i * size + x_i
position_index = cell_0 + cell_1 * size²
target_code = active ? target_id + 1 : 0
observation_index = target_code * size⁴ + position_index
```

Both agents receive this index. There are `(6 + 1) * 5⁴ = 4,375` observations.
The two-slot Q-table is 43,750 float32 values, or 175,000 uncompressed bytes.
The greedy action table is only 8,750 bytes. The latter is sufficient to reproduce
greedy actions exactly; ship Q-values only if the page displays values or margins.

The action table is computed from the source float32 Q-values with the declared
mask and tie rule. This export profile requires action availability to be
determined by the observation index. If it is not, an index-only action table is
not a valid representation. A stochastic execution policy would also require
probabilities or logits, not a table of argmax actions.

Coordinates remain logical: x increases right, y increases up. Canvas projection
flips y. Animal art rotation, subcell animation, and display resizing cannot alter
logical coordinates. This also aligns the scene with Python scientific plots.

### Scenario schema

A cooperative-grid scenario contains `id`, `positions[A,2]`, `target_id`, `active`,
and `steps`. These are the complete initial state for the deterministic demo.
Reference records contain initial state followed by each policy input, joint
action, next state, reward, and end flags. Keeping concrete states avoids a false
promise that Python and JavaScript seeds produce the same reset.

The current environment samples target/start positions at reset but transitions
deterministically thereafter. The browser can therefore cycle explicit scenarios
without a cross-language PRNG. For a later stochastic environment, define a shared
random-event contract or provide a recorded event stream for parity checks.
Ordinary stochastic simulation may use its own RNG, but cannot claim exact seeded
trajectory equivalence without a matching algorithm and draw order.

## Simulation clock and episode lifecycle

One tick reads the same pre-step observations for all agents, computes all
actions, then advances the world once. It never computes one agent's action from
another agent's partially advanced state.

Use a fixed simulation interval, initially 180 ms per logical step, with an
accumulator independent of `requestAnimationFrame`. Rendering interpolates from
the previous logical state to the current one. No velocity integration or
per-frame friction changes the environment. Pause the simulation while the page
is hidden; resuming does not run an unbounded backlog of steps.

When success, wrong-target failure, or timeout occurs, preserve and display the
terminal frame, pause for a duration measured in seconds, then reset to the next
scenario and clear policy carry. Success delay is not extra policy steps.

Default playback cycles scenarios without requiring mouse input. Pause, resume,
restart, scenario choice, and method choice are explicit controls. A perturbation
mode, if added, pauses playback, snaps a moved animal to a valid logical state,
clears recurrent carry, and resumes as a labeled intervention. A changed target
or layout must remain within the selected policy's observation/action schema;
choosing arbitrary grid sizes is not supported by a fixed tabular policy.

## Supported export profiles

| Profile | Python freeze | JavaScript requirement | Stage |
| --- | --- | --- | --- |
| Greedy table | Action lookup; optional original Q-values | Integer encoding and lookup | First |
| Feed-forward Q / actor MLP | Weights, biases, normalization, ordered feature schema | Explicit dense layers and named activations | After neural demo |
| Recurrent actor | Same plus recurrent architecture and initial carry | Exact recurrence and episode reset | Later |
| Planner | Dynamics model plus planner, or separately evaluated distilled actor | Explicit supported planning runtime | Deferred |

For a first MLP profile, restrict layers to dense + ReLU/tanh and define kernel
shape `[input, output]`, bias shape `[output]`, float32 weights, preprocessing,
normalization epsilon, output interpretation, and sampling/argmax behavior.
Numerical comparisons use a stated tolerance; float32 storage alone does not
make JavaScript arithmetic identical to JAX. Check action agreement on reference
inputs, especially near ties. Export support is a capability of a model and an
environment, not an automatic consequence of satisfying the Python API.

## Small handoff check

At export, compare the table policy's action choices on every finite table index
(cheap for this demo), and compare a handful of world trajectories across Python
and JavaScript. Include activation, collection, wrong target, and timeout states.
Reuse these records for one small runtime test. No retraining or visual screenshot
suite is required. Evaluate the exported policy representation; if conversion
changes actions, do not attach the original policy's scores to the new one.

The site build copies a selected bundle, runtime modules, renderer, and assets to
`site/`. The Pages workflow builds/uploads those static files without importing
the Python training stack. The library can run locally from any static server.
