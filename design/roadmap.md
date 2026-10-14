# Implementation sequence and showcase plan

This is a proposed execution order, not work completed in the current pass.
Implement one working slice at a time. Minimal tests protect interfaces; actual
learning runs answer experiment questions.

## Stage 1 — Rehouse the existing example

Introduce the small environment, model, agent, method, and transition records.
Move batching/reset into the online runner. Use `QTable` + `QLearning`, with
normalized duplicate-index updates and explicit truncation semantics. Separate
the current shared observation index from privileged simulator state.

Move the experiment into a public-API recipe, use unique run directories, and
evaluate through the same environment implementation. Preserve the previous
saved run as history. Changing the update and timeout semantics means this is a
new run, not a byte-for-byte regeneration of the old checkpoint.

Done when one short script can construct, train, freeze, and evaluate the current
task without importing an experiment module into library code. Use a tiny smoke
run first; a full demo run is a separate intentional action.

## Stage 2 — Close the browser handoff

Define and export `marlax.browser` v1. Split gallery code into runtime, environment
port, renderer, and page. Match discrete steps and episode endings, interpolate
display only, and cycle explicit scenarios. Extract the current mouse into the
asset structure before producing additional animal art.

Done when a static server runs the frozen policy, a small reference trajectory
matches Python, and changing the page/art does not invoke learning. Publishing
remains the existing GitHub Pages workflow with a static build/copy step.

## Stage 3 — Establish method interchangeability

Add SARSA and Expected SARSA using the same Q-table, observation schema, and world.
Add only the collector behavior required to preserve SARSA's actual next action.
Keep Q-learning, SARSA, and Expected SARSA as named methods with readable updates.

Done when the recipe selects a method without editing the environment or browser
runtime. A policy selector loads separate frozen artifacts. It does not pretend
to switch learning rules inside a running browser episode.

## Stage 4 — Prove model-based learning with a small example

Use a fully observed, small cooperative gate task with a joint Q-table and an
empirical joint transition model. Compare joint Q-learning and Dyna-Q under the
same observations and action factorization. The comparison asks whether planning
reduces real environment samples; report planning updates and wall time too.

This needs a `JointQAgent` specialization, not a hidden global-state read inside
an independent agent. Freeze Q-values for browser execution; the planning model
stays offline. Keep the state space small enough to inspect.

## Stage 5 — Add one neural representation and one new world

Add `MLPQ` + DQN with replay, target parameters, and optional neural dependencies.
Use a compact cooperative foraging world with feature observations. Add a narrow
MLP browser profile only after that model is useful in Python. Reuse discrete
movement and assets; do not introduce a physics engine to demonstrate a network.

Done when a user-defined model can train through a supported method, freeze, and
execute with the same feature/action definitions in Python and JavaScript.

## Stage 6 — Expand according to research needs

Add IPPO before MAPPO, then a recurrent actor if a partial-observation task needs
memory. MAPPO adds privileged critic features while keeping actor inputs local.
Trace-based TD, n-step returns, Double Q-learning, Monte Carlo methods, value
decomposition, communication, mixed rewards, and continuous control are candidates,
not a requirement to finish the first usable library.

Do not implement every method named here upfront. Eligibility traces introduce
per-trajectory trace state; replay methods and trajectory methods have different
data assumptions. Let concrete experiments justify each extension.

## Demo zoo

Names describe intended scenes; these are proposed experiments, not claims of
already trained animal behaviors or biologically realistic models.

| Demo | Situation and observation | Learning comparison | Visible behavior / metrics | Stage |
| --- | --- | --- | --- | --- |
| Mice rendezvous | Existing 5×5 center activation, two target choices; shared joint index | Q-learning, SARSA, Expected SARSA | Meet, reveal, choose the same target; success, steps, wrong-target and timeout fractions | 1–3 |
| Mice at the gate | Small deterministic two-agent world; both positions and gate state observed | Joint Q-learning vs Dyna-Q | One opens a gate while the other crosses; real samples to success, planning updates, time | 4 |
| Monkeys forage together | Small obstacle grid, two agents, cooperative food collection; fixed feature vector | DQN vs a random baseline; tabular only on a tractable matching variant | Coordinate routes and collection; return, success, collisions or blocked moves | 5 |
| Elephants carry together | Two agents must align actions to move a shared object; fixed grid actions | IPPO, then MAPPO with the same local actor inputs | Synchronized movement; delivered distance, success, coordination errors | 6 |
| Mixed-species scout and carrier | Two roles with distinct strength/sensing, common action schema; partial observations | Recurrent policy vs feed-forward baseline | Scout reveals resources, carrier transports; communication/observation ablation | Later |

Mouse, monkey, and elephant skins can be previewed on the first world without
training separate policies. A genuinely different task needs different dynamics
or observations, not just a new animal picture.

Each demo should autoplay meaningful scenarios, show the method and observation
assumptions, expose pause/restart/scenario selection, and offer a compact evaluation
summary. Interventions are optional, explicitly labeled, and resettable. The
default experience needs no dragging or manual setup.

## Initial experiment budget

For stage 1, begin with a smoke profile of tens of vector steps and a small batch.
Then try a demo profile around 12 million world transitions, approximately one
tenth of the current recipe's 122.88 million. This is a starting budget, not a
guarantee of learning; extend it only after inspecting the curve and outcomes.
Evaluate at a few checkpoints rather than every update. One seed is adequate for
pipeline development; method-comparison claims require repeated seeds and a
documented budget, performed outside routine tests.

## Design choices to revisit only when a concrete task requires it

| Decision now | Trigger for revisiting |
| --- | --- |
| Fixed agent axis and common discrete actions | First task with differing spaces or agent lifetimes |
| Separate parameters and pure apply | First model needing mutable inference statistics |
| Named methods and a few runner families | A real algorithm that cannot fit their data flow |
| Explicit supported browser profiles | A model/environment that warrants another runtime backend |
| SVG outline assets | A demo that materially needs richer animation |
| Single-process JAX training | Measured need for device distribution |

No compatibility system or broad test investment is needed to begin. The success
criterion is that a second environment and a second learning method require
small, local additions rather than a rewrite of the experiment or gallery.
