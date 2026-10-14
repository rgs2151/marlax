# Python API proposal

All examples below describe the proposed API. Nothing here is implemented by this
design change. Use `import marlax as mx`; public names use ordinary Python casing.

## The common user path

```python
import marlax as mx
from marlax.envs import CooperativeGrid
from marlax.models import QTable
from marlax.methods import QLearning

env = CooperativeGrid(size=5, num_agents=2, observation="joint_index")
params = env.default_params()
agent = mx.Agent(model=QTable(), sharing="independent", policy="epsilon_greedy")
method = QLearning(alpha=0.03, gamma=0.97, epsilon=(1.0, 0.02))

run = mx.train(
    env=env,
    params=params,
    agent=agent,
    method=method,
    seed=0,
    num_envs=2048,
    env_steps=12_288_000,
    output="runs/coop_grid_td/q_learning_seed_0",
)
report = mx.evaluate(run.policy, env, params, scenarios="all_starts")
```

`run.policy` is the frozen policy from the final checkpoint. `run` also exposes
`checkpoint_path`, `metrics_path`, and resolved configuration. Exhaustive
`all_starts` evaluation is supplied by this small environment's scenario provider;
it is not a promise that every environment can enumerate its state space.

Replacing `QLearning(...)` with `SARSA(...)` changes the update and the runner's
next-action handling. Replacing `QTable()` with a network requires a method such as
`DQN`, not just a different constructor under tabular Q-learning.

## Environment

The constructor contains static structure: grid size, agent count, observation
mode, action schema. `params` contains runtime numerical task settings: rewards,
time limit, transition probabilities. Shape-changing edits require reconstruction
and recompilation; equal-shape parameter values can vary across runs or worlds.

| Member | Contract |
| --- | --- |
| `env.spec` | Static `EnvSpec`: ID/version, ordered agent IDs, observation spec, action spec, global-state spec, reward semantics |
| `env.default_params()` | Numeric PyTree of task settings |
| `env.reset(key, params)` | Returns `(state, timestep)` for one world |
| `env.step(key, state, actions, params)` | Returns `(next_state, timestep)`; never resets |
| `env.observe(state, params)` | Returns `PolicyInput(obs, action_mask)` |
| `env.global_state(state, params)` | Returns privileged critic features, or `()` when unused |

`reset` and `step` construct their timestep using `observe` and `global_state`.
The state contains the world information needed for its dynamics. Observation
noise, if used later, is sampled in reset/step and stored in state so `observe`
remains deterministic. Rendering and tabular state enumeration are optional
external adapters, not methods every environment must implement.

### Shapes and records

The initial profile has fixed `A` agents, simultaneous decisions, a common discrete
action count `U`, and episode-level termination. Every slot acts every step. Mixed
species can use this profile when their numerical interfaces match.

| Value | Single-world shape / meaning |
| --- | --- |
| `actions` | `int32[A]`, one action per agent in `spec.agent_ids` order |
| `PolicyInput.obs` | PyTree with leading `[A]`; table indices `[A]` or features `[A, F]` |
| `PolicyInput.action_mask` | `bool[A, U]`; each row has at least one legal action |
| `TimeStep.policy_input` | Observation and availability at the returned state |
| `TimeStep.global_state` | Critic-only numeric features; no implicit actor access |
| `TimeStep.reward` | `float32[A]`; team reward is repeated, not multiplied by agent count |
| `TimeStep.terminated` | Scalar bool: task success/failure or intrinsic task horizon |
| `TimeStep.truncated` | Scalar bool: external collection time limit |
| `TimeStep.metrics` | Fixed-key numeric PyTree of environment events |

Reset returns zero reward, false end flags, and zero event metrics. Episode end is
`terminated | truncated`; when both causes occur on a step, termination determines
the bootstrap mask. The runner resets only after recording that step. Independently
dying agents, heterogeneous spaces, and turn-taking need a later contract extension;
do not silently squeeze them into these episode-level flags.

With batching, these leaves gain leading `[B]`; stored trajectories gain `[T, B]`.
For example, actions become `[T, B, A]`. Specs, strings, and schema identifiers
remain host-side metadata and do not travel through `scan`.

### Observation access

`joint_index` reproduces the prototype's shared information: both agents receive
an encoding of both positions and the revealed target. This is explicitly a
shared-observation policy, not a claim of decentralized local sensing. The hidden
target must remain absent before center activation.

`local_features` is a separate environment observation mode, with a distinct
observation schema. `global_state` may give a centralized critic more information,
but `Agent.act` receives only `policy_input`. Hidden state needed by a recurrent
actor is explicit carry; it cannot be replaced by reading simulator internals.

### Extending an environment

A new class supplies `spec`, `default_params`, `reset`, `step`, `observe`, and
`global_state`, using its own small state record. Subclassing `mx.Environment`
provides a recognizable interface but does not create a mutable simulator.

```python
class Foraging(mx.Environment):
    def step(self, key, state, actions, params):
        next_state, reward, events = advance_foraging(key, state, actions, params)
        timestep = mx.TimeStep(
            policy_input=self.observe(next_state, params),
            global_state=self.global_state(next_state, params),
            reward=reward,
            terminated=events.completed,
            truncated=next_state.steps >= params.time_limit,
            metrics={"food_collected": events.food_collected},
        )
        return next_state, timestep
```

This excerpt shows only the transition method; the environment author supplies
the remaining members and `advance_foraging`. Use a NamedTuple or registered
dataclass PyTree for numerical records. Do not create a Python object per animal
inside a compiled step.

### Time limits

Choose a meaning when defining a task. For a continuing task cut off for collection,
use `truncated=True` and bootstrap from the final observation. For a task where
failure to finish within a budget is itself the outcome, use `terminated=True`
and include remaining budget in observations. The first migration should declare
the cooperative grid limit an external truncation. This changes learning semantics
from the prototype, which currently zeros the bootstrap at every timeout; record
the new environment version and retrain before publishing.

## Model

The smallest useful contract is:

```python
variables = model.init(key, input_spec, output_spec)
carry = model.initial_carry()
outputs, next_carry = model.apply(variables, inputs, carry, key)
```

This describes one policy slot and one observation. The agent adds policy-slot
and world axes. A feed-forward model uses `carry=()`. Variables are a PyTree of
parameters and fixed inference statistics. Initially exclude mutable batch-normalization
updates; supporting those later requires an explicit state-output extension.

| Model | Input | Output |
| --- | --- | --- |
| `QTable` | Finite integer observation index | `q[U]` |
| `MLPQ` (later) | Feature vector | `q[U]` |
| `CategoricalActor` (later) | Feature vector | `logits[U]` |
| `ValueModel` (later) | Local or global critic features | Scalar value |

For the table, `input_spec` declares the number of valid indices. The model never
infers table size by reading an environment's private state. A custom network
implements the same init/apply contract or wraps a Flax module. Network width and
activation are model configuration; optimizer and target-network schedules are
method configuration.

Dynamics models have an explicit separate extension:
`dynamics.predict(variables, model_state, joint_actions, key)` returns predicted
next model state, rewards, and termination. Model state is a declared state or
belief representation. It is not automatically the environment's hidden state.

## Agent and frozen policy

`Agent` combines a model, parameter routing, and an action rule. For initial tasks,
`sharing="independent"` creates `A` parameter slots; `sharing="shared"` creates
one slot applied to each agent observation. Shared networks that need distinct
roles receive an explicit agent/role feature in the observation schema.

```python
policy_state = agent.init(key, env.spec)
carry = agent.initial_carry(env.spec)
decision = agent.act(policy_state, policy_input, carry, key, exploration=epsilon)
```

`decision` contains `actions[A]`, `carry`, and numeric `extras` such as log
probabilities or selected Q-values. Action masks apply before greedy selection or
sampling. Exact greedy ties select the lowest legal action index. The exploration
schedule belongs to the method; the policy implements the sampling rule.

`policy_state` contains model variables and any observation-normalization values.
The runner owns the random keys and per-world carry. It uses fresh keys per world
and agent and resets carry at every episode boundary. Recurrent carry is runtime
state, not learned weights.

`method.freeze(method_state)` returns a `FrozenPolicy`: immutable inference
configuration and variables, with `initial_carry()` and
`act(policy_input, carry, key)`. For the initial table export, freeze sets greedy
execution and removes exploration, optimizer, replay, and target-network state.
Recurrent policies still maintain carry during inference; "frozen" means weights
do not learn online.

## Method and runner

```python
method_state = method.init(key, agent, env.spec)
method_state, metrics = method.update(method_state, batch, key)
policy = method.freeze(method_state)
```

`method.init` initializes the agent's `policy_state` and method-specific learning
state. `method_state.policy_state` is the current behavior policy. A method also
declares `runner_kind`, `batch_kind`, and its model/action requirements. These are
small static descriptors, not a universal registry or runtime plugin system.

The method-state record retains the immutable agent/model description as static
PyTree metadata; only weights, optimizer values, and counters are numerical
leaves. This gives `update` and `freeze` access to the model without mutating the
method during `init`. Checkpoints serialize the constructor description alongside
arrays rather than serializing live Python callables.

The runner creates `Transition` values with current and next policy inputs,
actions, reward, termination/truncation flags, and current/next global features.
The next values always belong to the actual transition, never the subsequent
reset. Trajectory methods additionally store behavior log probabilities, values,
and recurrent carry as required. The batch kind is method-specific.

| Method | Model and data | Runner behavior |
| --- | --- | --- |
| `QLearning` | Table, transition batch, next legal-action max | Collect one batched step, update |
| `SARSA` | Table, transitions plus actual next behavior actions | Select and retain the next action before update |
| `ExpectedSARSA` | Table, next epsilon-greedy action probabilities | Use expectation under the declared behavior policy |
| `DynaQ` (later) | Small joint Q-table, empirical transition model | Real updates plus a configured number of planning updates |
| `DQN` (later) | Q-network, replay samples, target parameters | Collect, sample, optimize, update target |
| `IPPO` / `MAPPO` (later) | Actor + critic, ordered trajectories | Collect a rollout, compute advantages, optimize epochs |

For SARSA, a nonterminal next action sampled before the update is the action
executed next. At an external truncation, sample a bootstrap action from the final
observation; the next executed action instead belongs to a newly reset episode.
Never bootstrap from that reset observation. On-policy advantage recursions stop
at reset boundaries even when a truncation allows a final value bootstrap.

Initial tabular updates use the mean TD error per occupied `(policy_slot,
observation_index, action)` within a batch. Unvisited entries stay unchanged.
This explicitly prevents duplicate visits from scaling the effective alpha with
the number of parallel environments. It is a synchronous batched update, not
equivalent to a sequence of online scalar updates.

For Q-learning the target is `reward + gamma * (1 - terminated) * next_max_q`,
where the max includes only legal next actions. A truncation keeps this bootstrap
but ends the collected episode. SARSA substitutes the retained next action value;
Expected SARSA substitutes the expectation under its next behavior distribution.

`env_steps` counts world transitions, `vector_steps` counts batched calls,
`updates` counts learning updates, and `episodes` counts completed worlds.
Exploration schedules use `env_steps`. Store all four counters. A batched run's
actual transition count is the requested budget rounded up to a multiple of `B`.

## Model-based learning is a genuine separate path

The first model-based example should use small, fully observed joint states and
joint discrete actions. `JointQAgent` produces a single joint action code and
decodes it to `[A]`; its action count is `U ** A`. Its empirical dynamics model
conditions on both agents' actions, avoiding a misleading assumption that a
single animal's action determines the whole world's transition. Share the TD
update primitives, not the claim that this is independent decentralized learning.

Dyna-Q can freeze its learned Q policy without exporting the planning model.
A later online planner must export both dynamics and planner, or explicitly
distill to a smaller actor. Expensive planning does not become cheap merely
because learning has stopped. Distillation creates a new policy requiring its
own evaluation; it is never a silent export conversion.

## Extension cost

| User wants to add | Work required |
| --- | --- |
| New environment for Python research | Environment contract and a short experiment recipe |
| New function representation | Model contract; choose a compatible method |
| New learning rule | Method update/state, declared data requirements, freeze behavior |
| New on-policy collection semantics | A focused runner extension only if current runner cannot express it |
| New browser environment | Explicit JavaScript dynamics/observation port and scene adapter |
| New browser model family | Explicit exporter and inference implementation for that family |
| Another visual animal | Asset entry and scene binding; no learning changes |

Keep unsupported combinations explicit in documentation. Do not implement a
compatibility layer, generic graph compiler, or exception framework in this phase.
