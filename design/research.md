# Existing libraries and what MARLAX should borrow

Primary documentation and source inspected on 2026-10-14. This is a focused
architecture study, not a benchmark or an exhaustive survey. Upstream default
branches can change; links identify the inspected interfaces rather than pinning
dependencies for MARLAX.

## Closest references

| Project | Observed design | Recommendation for MARLAX |
| --- | --- | --- |
| [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/) | Multi-agent actions and results keyed by agent; simultaneous steps; separate termination and truncation; global state available for centralized training | Borrow the semantics and explicit agent identities. Use dense agent axes for the initial homogeneous JAX tasks. Add an adapter only when an actual external environment is needed. |
| [JaxMARL](https://github.com/bold-lab-ai/JaxMARL) | JAX multi-agent environments plus algorithm baselines; explicit keys and state; parallel actions; agent-keyed dictionaries | Closest overall technical reference. Borrow accelerator-native rollouts and clear environment/algorithm separation. MARLAX's emphasis is reusable composition plus a small browser handoff. |
| [Gymnax](https://github.com/RobertTLange/gymnax) | JAX environments with explicit state and parameters, vectorizable execution | Borrow single-world functions and explicit environment parameters. Let the runner own batch size. |
| [Mava](https://github.com/instadeepai/Mava) | Research-oriented JAX MARL implementations, environment wrappers, and scalable training systems | Borrow readable method implementations and wrapper boundaries. Defer distribution architecture and configuration infrastructure. |
| [RLax](https://github.com/google-deepmind/rlax) | JAX mathematical building blocks for RL, rather than complete training systems | Keep return/TD computations as small functions used by named methods. Consider reuse when it saves code; do not require it for one tabular update. |
| [Flax Linen](https://flax-linen.readthedocs.io/en/latest/quick_start.html) | Modules initialize variables and apply them explicitly | Use a thin `Model.init` / `Model.apply` boundary. Later neural models can wrap Linen; MARLAX should not implement its own neural-network framework. |

PettingZoo is the relevant multi-agent counterpart to the Gym-style interface the
user described. Its Parallel API fits simultaneous animal actions. Its separate
[AEC API](https://pettingzoo.farama.org/api/aec/) is relevant if turn-taking becomes
a real requirement; it is outside the first MARLAX contract.

## Important choices rather than direct copies

**Reset belongs outside the physical transition.** JaxMARL's inspected
[base environment](https://raw.githubusercontent.com/bold-lab-ai/JaxMARL/main/jaxmarl/environments/multi_agent_env.py)
automatically resets in public `step`, while `step_env` performs the raw transition.
MARLAX should make raw `step` public and move reset into the runner. This makes the
terminal world visible to evaluation, learning, and browser animation.

**Preserve the final observation.** The inspected Gymnax
[environment source](https://raw.githubusercontent.com/RobertTLange/gymnax/main/gymnax/environments/environment.py)
includes final-observation information when automatically resetting. MARLAX can
avoid needing that repair by storing the transition before the runner resets.

**Name the reason an episode ends.** Gymnasium distinguishes natural termination
from external truncation because their bootstrap targets differ. It also notes
that a finite task horizon must be observable to retain the Markov property.
MARLAX should record both flags and explicitly define its time-limit semantics.
See [Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

**Use a familiar class surface with functional execution.** JAX transformations
work with explicit state and pure functions. Immutable configuration objects can
provide methods, while arrays and evolving state travel through their inputs and
outputs. This is the appropriate interpretation of the requested `nn.Module`-like
ergonomics. See [JAX stateful computations](https://docs.jax.dev/en/latest/stateful-computations.html).

**Randomness is part of the interface.** JAX uses explicit random keys and key
splitting. The runner should separate initialization, environment, exploration,
learning, and evaluation streams. A matching integer seed in JavaScript does not
establish matching random trajectories. See [JAX pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html).

## What to build versus reuse

Build the small MARLAX contracts, animal situations, method composition, experiment
recipes, frozen artifact format, and JavaScript runtime. Start with JAX and the
existing tabular implementation. Add Optax and a thin Flax adapter with the first
neural method. External environment adapters and replay libraries should arrive
when a recipe needs them, rather than making every installation depend on them.

The proposed APIs in the other documents are MARLAX design recommendations. They
are not upstream APIs or claims that these projects already provide browser export.
