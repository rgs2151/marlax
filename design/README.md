# MARLAX design

The approved visual direction is documented in [Aesthetics](aesthetics.md).
It supersedes earlier artwork proposals; the architecture notes below retain
their own scope.

Status: proposed architecture, 2026-10-14. These documents describe future APIs;
the snippets are specifications, not APIs already available in the package.

Implementation update: stage 1 now provides the grid, QTable, Agent, the online
runner, Q-learning/SARSA/Expected SARSA, evaluation, and NPZ/JSON storage. The
implemented signatures and limitations are documented in
[the Sphinx documentation](../docs/source/index.rst). The JavaScript runtime,
neural models, and model-based methods below remain proposals. In particular, the
first implementation uses a finite-index environment spec and stateless policies;
it does not claim the broader future model/runner support.

MARLAX should make it easy to define a multi-agent situation, choose a learning
method and function representation, run accelerated experiments in JAX, and publish
a frozen behavior as a small, interactive browser simulation.

The first implementation should make the existing cooperative grid example fit
this structure. Additional methods and environments should exercise the boundaries
before the project grows a larger framework.

The current two-to-three-demo scope and JavaScript audit are in
[Interactive simulation gallery](simulation-gallery.md).

## Read in this order

| Document | Decision it supports |
| --- | --- |
| [Research](research.md) | What to borrow from PettingZoo, JaxMARL, Gymnax, Mava, RLax, and Flax |
| [Architecture](architecture.md) | Responsibilities, package layout, dependencies, and design principles |
| [Python API](python-api.md) | Environment, model, agent, method, and training contracts |
| [Experiments](experiments.md) | Running, evaluating, saving, and extending research recipes |
| [Browser handoff](browser-handoff.md) | Frozen artifacts and a separate JavaScript runtime |
| [Animals and assets](assets.md) | Reusable visual animals, themes, and simulation roles |
| [Roadmap and demos](roadmap.md) | Small implementation stages and concrete showcase experiments |

## Main decisions

1. Use a small class interface over pure JAX functions. Runtime state, weights,
   optimizer state, and random keys are explicit values, never hidden mutations.
2. Separate `Environment`, `Model`, `Agent`, and `Method`. An agent acts; a method
   learns; a model represents a function; an environment defines the world.
3. Write an environment for one world. The runner adds parallel worlds with
   `vmap` and time with `scan`.
4. Start with simultaneous, fixed-population, discrete-action cooperative tasks.
   Keep observation access explicit so local policies and centralized critics can
   be added without leaking privileged information into execution.
5. Offer named methods such as `QLearning`, `SARSA`, and later `DynaQ`, `DQN`,
   `IPPO`, and `MAPPO`. Document valid combinations instead of promising that any
   method works with any model or environment.
6. Make JavaScript its own inference/simulation library. A versioned data artifact
   connects it to Python; no Python runtime or training service is needed online.
7. Keep species artwork separate from policy identity and physical capabilities.
8. Keep tests small. No training convergence jobs, coverage targets, or broad
   compatibility matrix during this stage.

## What this proposal deliberately postpones

Distributed training infrastructure, plugin discovery, arbitrary graph export,
turn-based games, changing agent populations, heterogeneous action spaces,
continuous-control browser engines, and a comprehensive benchmark suite. These are
possible later extensions, not prerequisites for useful research now.

No implementation changes are included in this design pass. The existing saved
run and gallery remain the starting reference. The first milestone is one working
vertical slice, not scaffolding every proposed directory at once.
