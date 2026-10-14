# Demo candidates: believable multi-agent behaviour

Research and design, 14 October 2026. The cooperative grid is the user's existing
environment and remains demo one. Paired docking and orbit rendezvous were assistant
proposals, never implemented, and are rejected. The earlier three-panel concept
sheet is not the demo plan. Interaction is secondary; the next demos should work
as compelling recorded simulations and standalone SVG animations.

This document proposes candidates, not an approved selection or implementation.
The approved Vector / 84 assets and black/white/dark-red palette remain unchanged.

## What makes a candidate worth showing

The viewer should understand what the agents need from each other. Prefer pursuit,
shared physical work, yielding, or collective motion over several independent trips
to markers. Motion should follow the environment's actual dynamics: bounded speed,
acceleration, turning and contact where relevant. Rendering may interpolate sampled
poses but must not invent dynamics, collision avoidance or successful cooperation.

Realism here means physically coherent stylized behaviour, not a validated model of
animal biology. Keep the existing overhead animal/vehicle silhouettes; assign them
to roles that fit. Do not change the grid's trained rules to make it appear to be a
continuous physical simulation.

## Shortlist

| Candidate | Established reference | Why it could work here | Main cost / concern |
| --- | --- | --- | --- |
| Pursuit–evasion | [MPE2 Simple Tag](https://mpe2.farama.org/environments/simple_tag/), [JaxMARL MPE](https://jaxmarl.foersterlab.com/environments/mpe/) | A chase makes anticipation, interception and coordination visible | Avoid training two changing teams at once initially; point-agent physics is not an animal locomotion model |
| Cooperative transport | [VMAS transport](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/transport.py) | A shared object makes the effects of cooperation tangible | Requires meaningful contact dynamics and object rotation; highest physics cost of the recommended pair |
| Flocking around obstacles | [VMAS flocking](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/flocking.py) | Strong visual potential from a group separating and regrouping | A pretty flock can be a hand-written controller; learned behaviour must be distinguished from a baseline |
| Yielding in a narrow passage | [VMAS give_way](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/give_way.py) | A clear social decision: one agent waits or moves aside for another | Less visually varied; retain the bottleneck so it does not become the rejected docking task |
| Cooperative coverage | [MPE2 Simple Spread](https://mpe2.farama.org/environments/simple_spread/) | A straightforward established MARL example | Lower visual priority: reaching landmarks is close to the rejected destination-based design |

Ranking and visual suitability are design judgments. Sources establish the tasks;
the proposed casts, layouts and shot sequences below are our adaptations.

## Recommended next demo: pursuit–evasion

Simple Tag pits slower pursuers against a faster evader with blocking obstacles.
JaxMARL also documents a cooperative variant with heuristic-controlled prey. These
provide concrete reference tasks and an existing JAX implementation to investigate.
Sources: [MPE2 task](https://mpe2.farama.org/environments/simple_tag/),
[JaxMARL variants](https://jaxmarl.foersterlab.com/environments/mpe/).

Proposed presentation: three foxes and one mouse in an open enclosure with two or
three solid obstacles. Use the existing fox and mouse assets. Reserve dark red for
a capture/contact indicator, not team colouring. Give the chase room to develop;
show changing pursuit angles and an interception attempt, not an instant reset.

Start with learned pursuers and an explicitly documented fixed evader controller.
This makes the pursuers' learning problem stationary. Training the evader as well
is a separate experiment, not a prerequisite. A fixed controller is not presented
as a learned animal. If we later constrain turn rate or alter capture conditions,
that defines a modified task requiring training under those same rules.

The desired clip has an approach, a route change around an obstacle, and either
capture or escape. Flanking is a behaviour to look for in actual rollouts, not an
animation instruction or an outcome we can promise before training. Use varied
initial states and several recorded episodes rather than injecting random movement
into the renderer to manufacture uncertainty.

Why first: immediately legible, fits the existing animals, and has a concrete JAX
reference. It offers a different coordination problem from the cooperative grid.

## Recommended third demo: cooperative object transport

VMAS transport rewards agents for moving heavy packages toward a goal; moving the
load effectively requires collaboration. Reference:
[VMAS transport](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/transport.py).

Proposed presentation: two or three rovers push one long crate onto a loading pad.
The rover is an existing approved asset; the crate and floor layout would need
simple matching SVG assets. Vehicles make the pushing mechanics easy to interpret
without claiming realistic elephant or monkey biomechanics.

The important visible event is a change in the load's motion when agents make
contact from different positions. The crate must have position, orientation,
linear/angular velocity and contact response in the reference simulation. Its
motion cannot be a scripted path that starts when two agents get close. Start on
an open floor; do not add a narrow doorway until basic transport is learned.

A useful recording shows approach, asymmetric contact, rotation or drift, a
correction, and delivery. Those are selection criteria for genuine trajectories,
not a choreography to hard-code. Include an imperfect rollout if it helps explain
the task, without concealing failures behind endless successful loops.

Why third: it adds shared physical work, distinct from rendezvous and chasing.
The tradeoff is physics and training effort. VMAS is implemented in PyTorch, so it
is a task/physics reference here, not a decision to replace the JAX constraint.
Source: [official VMAS repository](https://github.com/proroklab/VectorizedMultiAgentSimulator).

## Alternative if collective animal motion matters more: flocking

VMAS includes a flocking task combining group cohesion, movement and collision
avoidance. Reference:
[VMAS flocking](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/flocking.py).

Proposed adaptation: six to eight animals move through an open obstacle field.
Look for compression near an obstruction and regrouping afterward. Keep one
coherent species group; do not mix arbitrary animals just to display every asset.
The current mouse artwork could support a stylized group, but this would not be a
claim about biological mouse flocking. Species-specific schooling/herding would
need its own design and suitable assets.

This is my strongest alternative to transport for visual appeal, but weaker for
explaining why RL is needed. Compare a learned controller with a simple rule-based
baseline privately; do not label a scripted flock as learned MARL. A training clip
must demonstrate behaviour actually produced by its policy.

## Two other references, not the recommendation

VMAS give_way uses a constrained corridor with a passing space. Our adaptation
could use two rovers, where one waits in the side bay. The bottleneck creates a
meaningful inter-agent dependency. It is a useful simpler fallback if transport
proves too costly, but not just new artwork for paired docking.
[Reference](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/give_way.py).

Simple Spread is a useful established coverage benchmark, but the destination
assignment story is too close to the rejected navigation proposal for this small
gallery. Multiwalker was also reviewed: its articulated walking and balancing
would require a different visual embodiment from the frozen top-down pieces, so
it is not a good next demo.
[Spread](https://mpe2.farama.org/environments/simple_spread/),
[Multiwalker](https://pettingzoo.farama.org/environments/sisl/multiwalker/).

## JAX and the current package boundary

An existing JAX benchmark is not automatically compatible with MARLAX's current
finite-index observations and tabular model. The continuous-state candidates need
an explicit training approach: a suitable discretization demonstrated to work, an
experiment integration with an existing JAX learner, or a separately designed
vector-observation/model extension. Do not squeeze a large joint physical state
into a tiny Q-table just to avoid admitting this boundary.

JaxMARL's MPE documentation describes vector observations and discrete or continuous
actions. It is the most direct implementation reference for pursuit. VMAS provides
useful physical task examples but uses a different numerical stack.
[JaxMARL](https://jaxmarl.foersterlab.com/environments/mpe/),
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator).

No new dependencies, package APIs, learners, environments or training runs are
introduced in this design pass. Select the task first, then make the smallest
explicit training decision needed for it.

## Recorded demonstrations first

For the new environments, run the authoritative simulation offline and export
positions, headings, object poses and task events at a stated timestep. JavaScript
and SVG display the recorded result. They need no copy of the physics solver or
neural inference implementation unless live simulation later becomes a requirement.
This removes browser physics parity as a prerequisite for showing a good demo.

The existing grid recorder/view split is a useful start. A new recording should
retain explicit body heading, especially for rotating crates; deriving heading
only from displacement is insufficient. Sample quickly enough to preserve contact
and turning, and interpolate wrapped angles correctly. Keep source trajectories
and rollout seeds so a clip can be regenerated. Do not smooth a path through an
obstacle, move a body before contact, or retime each agent independently.

Recommended gallery scope: **the existing cooperative grid, pursuit–evasion, and
cooperative transport**. Flocking is the alternative third demo if collective
animal motion is the stronger priority. These are recommendations awaiting task
selection, not additions to the live gallery.
